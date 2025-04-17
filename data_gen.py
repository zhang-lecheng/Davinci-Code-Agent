# -*- coding: utf-8 -*-
import os
import copy
import random
import datetime
import re
import math
import heapq
import csv # For writing TSV files
from tqdm import tqdm # For progress bar

# --- PyTorch / Model Imports ---
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn import Transformer
except ImportError:
    print("错误：缺少 PyTorch 库。请使用 'pip install torch' 进行安装。")
    exit()

# --- 环境导入 ---
try:
    from env import DaVinciCodeGameEnvironment # 从 env.py 导入
except ImportError:
    print("错误：无法找到环境文件 'env.py' 或 'DaVinciCodeGameEnvironment' 类。请确保文件存在且路径正确。")
    exit()

# --- 全局常量 ---
HUMAN_PLAYER_ID = 0 # 在自对战中，可以认为一个是 0，另一个是 1
OPPONENT_PLAYER_ID = 1
LOG_FILENAME = "davinci_self_play_run.log" # General run log
OUTPUT_DIR = "output_histories"
GAMES_PER_FILE = 1000
TOTAL_GAMES = 100000 # 目标生成 10 万局
MAX_TURNS_PER_GAME = 200 # 每局最大回合数，防止死循环

# --- Vocabulary and Tokenizer Definitions ---
# (与 Streamlit 版本一致)
PAD_TOKEN = "[PAD]"; SOS_TOKEN = "[SOS]"; EOS_TOKEN = "[EOS]"; UNK_TOKEN = "[UNK]"
KNOWN_TOKEN = "[KNOWN]"; CARD_TOKENS = [f"B{i}" for i in range(12)] + [f"W{i}" for i in range(12)] + ["B-", "W-"]
NUMBER_TOKENS = [str(i) for i in range(12)]
HIDDEN_MARKERS = ["B?", "W?"]
STRUCTURAL_TOKENS = ["your", "hand:", "opponent's", ";", "[", "]", ":", "!", "?", " "]
MODEL_VOCAB_V3_REVISED = sorted(list(set(([PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, KNOWN_TOKEN] + CARD_TOKENS + NUMBER_TOKENS + HIDDEN_MARKERS + STRUCTURAL_TOKENS))))
token_to_idx = {token: idx for idx, token in enumerate(MODEL_VOCAB_V3_REVISED)}
idx_to_token = {idx: token for token, idx in token_to_idx.items()}
VOCAB_SIZE = len(MODEL_VOCAB_V3_REVISED)
PAD_IDX = token_to_idx[PAD_TOKEN]; SOS_IDX = token_to_idx[SOS_TOKEN]; EOS_IDX = token_to_idx[EOS_TOKEN]
KNOWN_IDX = token_to_idx[KNOWN_TOKEN]; UNK_IDX = token_to_idx[UNK_TOKEN]

def tokenize_v3_revised(input_str, target_list):
    escaped_tokens = [re.escape(token) for token in MODEL_VOCAB_V3_REVISED if token not in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, KNOWN_TOKEN]]
    escaped_tokens.sort(key=len, reverse=True)
    pattern_parts = []
    if 'B?' in token_to_idx: pattern_parts.append(re.escape('B?'))
    if 'W?' in token_to_idx: pattern_parts.append(re.escape('W?'))
    if ':!' in token_to_idx: pattern_parts.append(re.escape(':!'))
    elif ':' in token_to_idx and '!' in token_to_idx: pass
    for token in escaped_tokens:
        if token not in [re.escape('B?'), re.escape('W?'), re.escape(':!')]:
             if token == re.escape('?') and ('B?' in token_to_idx or 'W?' in token_to_idx): continue
             if token == re.escape(':') and ':!' in token_to_idx: continue
             if token == re.escape('!') and ':!' in token_to_idx: continue
             pattern_parts.append(token)
    pattern = r'(' + '|'.join(pattern_parts) + r'|\s+)'
    found_tokens = [match[0] for match in re.findall(pattern, input_str) if not match[0].isspace()]
    input_token_indices = [token_to_idx.get(token, UNK_IDX) for token in found_tokens]
    target_token_indices = [SOS_IDX] + [token_to_idx.get(token, UNK_IDX) for token in target_list if token != PAD_TOKEN]
    return torch.tensor(input_token_indices, dtype=torch.long), torch.tensor(target_token_indices, dtype=torch.long)

tokenize = tokenize_v3_revised

# --- Transformer Model Definition ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__(); self.dropout = nn.Dropout(p=dropout); position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)); pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term); pe[:, 0, 1::2] = torch.cos(position * div_term); self.register_buffer('pe', pe)
    def forward(self, x): seq_len = x.size(1); pe_to_add = self.pe[:seq_len].squeeze(1); x = x + pe_to_add; return self.dropout(x)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int, emb_size: int, nhead: int, src_vocab_size: int, tgt_vocab_size: int, dim_feedforward: int = 512, dropout: float = 0.1, max_seq_len: int = 500):
        super().__init__(); self.transformer = Transformer(d_model=emb_size, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.generator = nn.Linear(emb_size, tgt_vocab_size); self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size); self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size); self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout, max_len=max_seq_len)
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
    def forward(self, src: torch.Tensor, trg: torch.Tensor, src_padding_mask: torch.Tensor, tgt_padding_mask: torch.Tensor, memory_key_padding_mask: torch.Tensor, tgt_mask: torch.Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src)); tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg)); outs = self.transformer(src_emb, tgt_emb, src_mask=None, tgt_mask=tgt_mask, memory_mask=None, src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=memory_key_padding_mask); return self.generator(outs)
    def encode(self, src: torch.Tensor, src_padding_mask: torch.Tensor): src_emb = self.positional_encoding(self.src_tok_emb(src)); return self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)
    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor, tgt_padding_mask: torch.Tensor, memory_key_padding_mask: torch.Tensor): tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt)); return self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_mask=None, tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=memory_key_padding_mask)

# --- Model Loading Function ---
def load_model(model_path, device):
    """加载训练好的模型权重"""
    print(f"正在加载模型权重从: {model_path}")
    # 定义模型参数 (应与训练时一致)
    num_encoder_layers=3; num_decoder_layers=3; emb_size=256; nhead=8
    dim_feedforward=512; dropout=0.1; max_seq_len=500
    # 初始化模型结构
    model = Seq2SeqTransformer(num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                               emb_size=emb_size, nhead=nhead, src_vocab_size=VOCAB_SIZE,
                               tgt_vocab_size=VOCAB_SIZE, dim_feedforward=dim_feedforward,
                               dropout=dropout, max_seq_len=max_seq_len).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载模型权重: {model_path}")
        model.eval() # 设置为评估模式
        return model
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 '{model_path}'。将使用随机初始化的模型！")
        model.eval() # 仍然设置为评估模式
        return model # 返回未训练的模型
    except Exception as e:
        print(f"加载模型权重 '{model_path}' 时出错: {e}。将使用随机初始化的模型！")
        model.eval()
        return model

# --- 辅助函数 ---
def construct_model_input_string(env, current_state, current_player_id):
     """
     为指定玩家构建模型输入字符串 (不含!guesses)。
     'your hand' 是 current_player_id 的手牌。
     'opponent's hand' 是另一个玩家的可见手牌。
     """
     other_player_id = 1 - current_player_id

     # 当前玩家的手牌 (your hand)
     my_hand = []
     if hasattr(env, 'hands') and current_player_id < len(env.hands) and isinstance(env.hands[current_player_id], list):
         my_hand = env.hands[current_player_id]
     drawn_card = current_state.get('drawn_card') # 只有当前玩家知道自己抽到的牌
     full_my_hand = sorted(my_hand + ([drawn_card] if drawn_card else []), key=env.card_value)
     your_hand_str = "".join([f"[{card}]" for card in full_my_hand])

     # 对手玩家的手牌 (opponent's hand)
     other_hand = []; other_revealed = []; opponent_hand_parts = []
     if hasattr(env, 'hands') and other_player_id < len(env.hands) and isinstance(env.hands[other_player_id], list):
         other_hand = env.hands[other_player_id]
     if hasattr(env, 'revealed') and other_player_id < len(env.revealed) and isinstance(env.revealed[other_player_id], list):
         other_revealed = env.revealed[other_player_id]

     if len(other_hand) == len(other_revealed):
         for i, card in enumerate(other_hand):
             if i < len(other_revealed):
                 if other_revealed[i]: opponent_hand_parts.append(f"[{card}]")
                 else: opponent_hand_parts.append(f"[{card[0]}?]") # 仅显示颜色
             else: opponent_hand_parts.append("[?]")
     else: opponent_hand_parts = ["[?]" for _ in other_hand]

     opponent_hand_str = "".join(opponent_hand_parts)
     input_str = f"your hand: {your_hand_str}; opponent's hand: {opponent_hand_str}"
     return input_str

# --- AI 动作决策函数 ---
def get_model_action(env, current_state, model_components):
    """使用模型为当前玩家决定下一个动作。"""
    model, tokenizer, vocab_info, device = model_components
    current_player_id = current_state.get('current_player')
    other_player_id = 1 - current_player_id

    # 1. 构建输入
    input_str = construct_model_input_string(env, current_state, current_player_id)
    # print(f"DEBUG (Player {current_player_id}): Model Input: {input_str}") # 减少打印

    # 2. 模型推理 (Single Best Guess)
    best_guess = None
    try:
        model.eval()
        src_tensor, _ = tokenizer(input_str, ["dummy"])
        src_tensor = src_tensor.unsqueeze(0).to(device)
        src_padding_mask = (src_tensor == vocab_info["PAD_IDX"]).to(device)

        # 解析 *对手* 的手牌表示以进行推理 (因为模型要预测的是对手的隐藏牌)
        opp_hand_part_match = re.search(r"opponent's hand: (.*)", input_str)
        hidden_indices = []; opponent_hand_len = 0
        if opp_hand_part_match:
             opp_hand_part = opp_hand_part_match.group(1)
             opp_cards_repr = opp_hand_part.replace('[', '').split(']')[:-1]
             opponent_hand_len = len(opp_cards_repr)
             hidden_indices = [idx for idx, card_repr in enumerate(opp_cards_repr) if '?' in card_repr]

        confident_guesses = []
        if opponent_hand_len > 0 and hidden_indices:
             with torch.no_grad():
                 memory = model.encode(src_tensor, src_padding_mask)
                 memory_key_padding_mask = src_padding_mask
                 tgt_input_ids = [vocab_info["SOS_IDX"]]
                 for i in range(opponent_hand_len):
                      tgt_tensor = torch.LongTensor([tgt_input_ids]).to(device)
                      tgt_padding_mask = (tgt_tensor == vocab_info["PAD_IDX"]).to(device)
                      tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_tensor.size(1), device=device)
                      decoder_output = model.decode(tgt_tensor, memory, tgt_mask, tgt_padding_mask, memory_key_padding_mask)
                      step_logits = model.generator(decoder_output[:, -1, :])
                      step_log_probs = F.log_softmax(step_logits, dim=-1)
                      top_log_prob, top_idx = torch.topk(step_log_probs, 1, dim=-1)
                      predicted_id = top_idx.item(); prediction_score = top_log_prob.item()
                      predicted_token = vocab_info["idx_to_token"].get(predicted_id, vocab_info["UNK_TOKEN"])
                      if i in hidden_indices:
                           if predicted_token in vocab_info["CARD_TOKENS"]:
                               confident_guesses.append({"score": prediction_score, "position": i, "guess": predicted_token})
                      tgt_input_ids.append(predicted_id)
                      # if predicted_id == vocab_info["EOS_IDX"]: break

        if confident_guesses:
             confident_guesses.sort(key=lambda x: x["score"], reverse=True)
             best_guess = confident_guesses[0]
             # print(f"DEBUG (Player {current_player_id}): Model Best Guess: {best_guess}") # 减少打印

    except Exception as e:
        print(f"警告: 玩家 {current_player_id} 模型推理失败: {e}")
        best_guess = None

    # 3. 决定动作
    opponent_action = None
    drawn_card = current_state.get('drawn_card')
    can_place = drawn_card is not None and ('place',) in current_state.get('legal_actions', [])
    # 检查 *对手* 是否还有隐藏牌可供猜测
    other_revealed = env.revealed[other_player_id] if hasattr(env, 'revealed') and other_player_id < len(env.revealed) else []
    can_guess = any(not r for r in other_revealed) if isinstance(other_revealed, list) else False

    if best_guess:
        pos = best_guess['position']; card = best_guess['guess']
        # 检查推荐的猜测位置是否有效 (在对手手牌范围内且未揭示)
        if pos < len(env.hands[other_player_id]) and not env.revealed[other_player_id][pos]:
             opponent_action = ('guess', pos, card)
        else: # 推荐位置无效
             if can_place: opponent_action = ('place',)
             elif can_guess: # 不能放置但还能猜，随机猜
                  hidden_indices_other = [i for i, r in enumerate(other_revealed) if not r]
                  target_index = random.choice(hidden_indices_other)
                  possible_cards = CARD_TOKENS; random_guess_card = random.choice(possible_cards)
                  opponent_action = ('guess', target_index, random_guess_card)
             else: opponent_action = None # 无法操作
    # 模型无建议时的回退
    elif can_place: opponent_action = ('place',)
    elif can_guess: # 不能放置但还能猜，随机猜
         hidden_indices_other = [i for i, r in enumerate(other_revealed) if not r]
         target_index = random.choice(hidden_indices_other)
         possible_cards = CARD_TOKENS; random_guess_card = random.choice(possible_cards)
         opponent_action = ('guess', target_index, random_guess_card)
    else: opponent_action = None # 无法操作

    return opponent_action, best_guess # 返回动作和模型猜测信息（用于后续判断）


# --- 主模拟脚本 ---
if __name__ == "__main__":
    print("--- 开始 DaVinci Code AI 自对战模拟 ---")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型 (只加载一次)
    model = load_model('./model.pth', device)
    tokenizer_func = tokenize
    vocab_info = { "SOS_IDX": SOS_IDX, "PAD_IDX": PAD_IDX, "EOS_IDX": EOS_IDX,
                   "idx_to_token": idx_to_token, "token_to_idx": token_to_idx,
                   "CARD_TOKENS": CARD_TOKENS, "UNK_TOKEN": UNK_TOKEN, "VOCAB_SIZE": VOCAB_SIZE }
    model_components = (model, tokenizer_func, vocab_info, device)

    # 初始化环境
    env = DaVinciCodeGameEnvironment()

    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出目录: {OUTPUT_DIR}")

    # 初始化文件和计数器
    file_index = 0
    game_count_in_file = 0
    current_file_path = ""
    tsv_writer = None
    file_handle = None
    timed_out_games = 0

    print(f"目标生成游戏数量: {TOTAL_GAMES}")
    print(f"每文件保存游戏数量: {GAMES_PER_FILE}")
    print(f"每局最大回合数: {MAX_TURNS_PER_GAME}")

    # --- 主循环 ---
    for game_id in tqdm(range(1, TOTAL_GAMES + 1), desc="生成游戏数据"):

        # --- 文件管理 ---
        if game_count_in_file == 0: # 需要打开新文件
            if file_handle: file_handle.close() # 关闭上一个文件
            file_index += 1
            current_file_path = os.path.join(OUTPUT_DIR, f"game_histories_{file_index}.tsv")
            print(f"\n开始写入文件: {current_file_path}")
            try:
                # 使用 'newline=""' 避免空行，指定 utf-8 编码
                file_handle = open(current_file_path, 'w', newline='', encoding='utf-8')
                tsv_writer = csv.writer(file_handle, delimiter='\t')
                # 写入表头
                tsv_writer.writerow([
                    "game_id", "turn", "player_id",
                    "input_state_str", "action_str", "reward", "done",
                    "best_guess_pos", "best_guess_card", "best_guess_score" # 添加模型猜测信息
                ])
            except IOError as e:
                print(f"错误：无法打开或写入文件 {current_file_path}: {e}")
                break # 停止模拟

        # --- 单局游戏模拟 ---
        state = env.reset()
        done = False
        turn_count = 0
        game_timed_out = False
        game_stuck = False

        while not done and turn_count < MAX_TURNS_PER_GAME:
            turn_count += 1
            current_player = state.get('current_player')
            if current_player is None: # 状态异常
                 print(f"错误：游戏 {game_id} 回合 {turn_count} 状态无效 (无当前玩家)。")
                 break

            # 获取当前状态字符串 (行动前的状态)
            state_str = construct_model_input_string(env, state, current_player)

            # 获取模型动作
            action, best_guess_info = get_model_action(env, state, model_components)

            # 记录模型猜测信息（即使动作是放置或其他）
            guess_pos = best_guess_info['position'] if best_guess_info else -1
            guess_card = best_guess_info['guess'] if best_guess_info else "N/A"
            guess_score = best_guess_info['score'] if best_guess_info else float('nan')


            if action is None:
                # AI 卡住，无法做出有效动作
                print(f"警告：游戏 {game_id} 回合 {turn_count}，玩家 {current_player} 卡住，无法行动。")
                game_stuck = True
                break # 结束本局游戏

            # 执行动作
            try:
                next_state, reward, done, info = env.step(action)

                # 记录这一步的数据
                row_data = [
                    game_id, turn_count, current_player,
                    state_str, str(action), f"{reward:.4f}", str(done),
                    guess_pos, guess_card, f"{guess_score:.4f}"
                ]
                tsv_writer.writerow(row_data)

                # --- 处理猜对后的逻辑 (根据分数决定是否放置) ---
                # 注意：这里的 correct_guess 需要重新判断，因为 get_model_action 不返回它
                if action[0] == 'guess':
                    # 检查上一步猜测是否真的正确 (通过比较环境状态变化)
                    # 这是一个简化的检查，可能不完全准确，依赖环境日志或更复杂的状态比较更好
                    # 假设：如果对手牌被揭示了，说明猜对了
                    other_player = 1 - current_player
                    pos_guessed = action[1]
                    revealed_after = next_state['opponent_revealed'] if current_player == HUMAN_PLAYER_ID else next_state['revealed_self'] # 获取对方的揭示状态
                    was_correct_this_step = revealed_after[pos_guessed] and not state['opponent_revealed'][pos_guessed] if current_player == HUMAN_PLAYER_ID else revealed_after[pos_guessed] and not state['revealed_self'][pos_guessed]

                    if was_correct_this_step and not done:
                        should_place_immediately = False
                        if best_guess_info is not None:
                            guess_score_from_log = best_guess_info.get('score', -float('inf'))
                            score_threshold = -1.2 # 与 UI 中一致的示例阈值
                            if guess_score_from_log < score_threshold:
                                should_place_immediately = True
                        # else: # 随机猜测正确，默认不放置

                        if should_place_immediately:
                            place_action = ('place',)
                            # 检查放置是否合法
                            if place_action in next_state.get('legal_actions', []):
                                state_before_place = next_state # 记录放置前的状态
                                state_str_before_place = construct_model_input_string(env, state_before_place, current_player) # 放置动作对应的输入状态

                                next_state_after_place, reward_place, done_after_place, info_place = env.step(place_action)
                                # 记录放置这一步的数据
                                row_data_place = [
                                    game_id, turn_count + 0.5, current_player, # 使用 .5 表示同一回合的第二步
                                    state_str_before_place, str(place_action), f"{reward_place:.4f}", str(done_after_place),
                                    -1, "N/A", float('nan') # 放置动作没有模型猜测
                                ]
                                tsv_writer.writerow(row_data_place)
                                next_state = next_state_after_place # 更新状态
                                done = done_after_place # 更新结束标志
                            # else: pass # 无法放置，回合已在 env.step 中切换

                # 更新状态以进行下一轮循环
                state = next_state

            except Exception as e:
                print(f"错误：游戏 {game_id} 回合 {turn_count} 执行动作 {action} 时出错: {e}")
                done = True # 出现错误时强制结束本局

        # --- 单局游戏结束 ---
        if turn_count >= MAX_TURNS_PER_GAME:
            game_timed_out = True
            timed_out_games += 1
            print(f"警告：游戏 {game_id} 达到最大回合数 {MAX_TURNS_PER_GAME}，强制结束。")

        # 记录游戏结束状态（可选，如果需要的话）
        # final_winner = env.winner if hasattr(env, 'winner') else state.get('winner', 'N/A')
        # final_row = [game_id, turn_count, "GAME_END", f"Winner: {final_winner}", str(game_timed_out), str(game_stuck), "", "", "", ""]
        # tsv_writer.writerow(final_row)

        game_count_in_file += 1
        if game_count_in_file >= GAMES_PER_FILE:
            game_count_in_file = 0 # 重置计数器，将在下一次循环开始时创建新文件

    # --- 主循环结束 ---
    if file_handle: file_handle.close() # 关闭最后一个文件
    print(f"\n--- 模拟完成 ---")
    print(f"总共模拟游戏局数: {TOTAL_GAMES}")
    print(f"超时游戏局数 (超过 {MAX_TURNS_PER_GAME} 回合): {timed_out_games}")
    print(f"数据已保存到目录: {OUTPUT_DIR}")
