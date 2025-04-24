from operator import is_
import streamlit as st
from toolz import last

# --- 页面配置 ---
st.set_page_config(layout="wide", page_title="达芬奇密码")

# --- 必要的导入 ---
import copy
import os
import random
import datetime
import re
import math
import heapq

# --- PyTorch / Model Imports ---
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn import Transformer
except ImportError:
    st.error("错误：缺少 PyTorch 库。请使用 'pip install torch' 进行安装。")
    st.stop()

# --- 环境导入 ---
try:
    from env import DaVinciCodeGameEnvironment # 从 env.py 导入
except ImportError:
    st.error("错误：无法找到环境文件 'env.py' 或 'DaVinciCodeGameEnvironment' 类。请确保文件存在且路径正确。")
    st.stop()

import json
import os
from openai import OpenAI 
# 设置 OpenAI API 密钥（你也可以将其存储在环境变量中）

client = OpenAI(                # 可以不传 api_key；SDK 会自己读环境变量 OPENAI_API_KEY
    api_key=
      # ← 如果你喜欢也可以硬写
)

# --- 样式定义 ---
st.markdown("""
<style>
    /* 卡牌基础样式 (HTML) */
    .card {
        border: 1px solid #cccccc;
        border-radius: 8px;
        padding: 15px 10px;
        margin: 5px;
        min-width: 60px;
        height: 90px;
        display: inline-flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        font-size: 1.5em;
        font-weight: bold;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        cursor: default;
        line-height: 1.2;
    }
    /* 黑色卡牌 (HTML) */
    .black-card {
        background-color: #333333;
        color: #ffffff;
        border: 2px solid #555555;
    }
    /* 白色卡牌 (HTML) */
    .white-card {
        background-color: #ffffff;
        color: #333333;
        border: 2px solid #cccccc;
    }
    /* 隐藏的对手卡牌样式 (问号) - 用于HTML卡牌 */
    .hidden-opponent-html {
        font-size: 2em;
    }
    /* 玩家自己的隐藏卡牌样式 (HTML) */
    .hidden-self {
        opacity: 0.8;
        border-style: dashed;
    }
    /* 已揭示标签样式 */
    .revealed-caption {
        font-size: 0.7em;
        text-align: center;
        margin-top: 2px;
        color: #555; /* 标签颜色 */
    }

    /* Streamlit 按钮的基本形状和大小调整 */
    div[data-testid="stButton"] > button {
        border-radius: 8px !important;
        padding: 5px !important;
        margin: 5px !important;
        width: 70px !important;
        height: 95px !important;
        display: inline-flex !important;
        justify-content: center !important;
        align-items: center !important;
        text-align: center !important;
        font-size: 1.6em !important;
        font-weight: bold !important;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1) !important;
        line-height: 1.2 !important;
        transition: background-color 0.2s ease !important;
    }

    /* 包裹按钮的 div 样式 */
    .black-card-button-wrapper, .white-card-button-wrapper {
        display: inline-block;
        margin: 0;
        padding: 0;
        line-height: 0;
    }

    /* 特定按钮颜色样式 */
    .black-card-button-wrapper button {
        background-color: #333333 !important;
        color: #ffffff !important;
        border: 2px solid #555555 !important;
    }
    .white-card-button-wrapper button {
        background-color: #ffffff !important;
        color: #333333 !important;
        border: 2px solid #cccccc !important;
    }
    .black-card-button-wrapper button:hover {
        background-color: #555555 !important;
        border-color: #777777 !important;
    }
     .white-card-button-wrapper button:hover {
        background-color: #e9ecef !important;
        border-color: #bbbbbb !important;
    }

    /* 猜测输入区域 */
    .guess-container {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin-top: 15px;
        background-color: #f9f9f9;
    }
    /* 用于显示抽到牌的容器 */
    .drawn-card-display {
         margin-top: 15px;
         padding: 10px;
         border: 1px dashed #ccc;
         border-radius: 5px;
         display: flex; /* 使用 flex 布局 */
         flex-direction: column; /* 垂直排列 */
         align-items: center; /* 水平居中 */
    }
</style>
""", unsafe_allow_html=True)


# --- 全局常量 ---
HUMAN_PLAYER_ID = 0
OPPONENT_PLAYER_ID = 1
LOG_FILENAME = "davinci_game_log.txt"

# --- Vocabulary and Tokenizer Definitions (来自用户提供) ---
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
    """使用 V3 Revised 词汇表对输入字符串和目标列表进行分词"""
    escaped_tokens = [re.escape(token) for token in MODEL_VOCAB_V3_REVISED if token not in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, KNOWN_TOKEN]]
    escaped_tokens.sort(key=len, reverse=True)
    pattern_parts = []
    if 'B?' in token_to_idx: pattern_parts.append(re.escape('B?'))
    if 'W?' in token_to_idx: pattern_parts.append(re.escape('W?'))
    for token in escaped_tokens:
        if token not in [re.escape('B?'), re.escape('W?')]:
             if token == re.escape('?') and ('B?' in token_to_idx or 'W?' in token_to_idx): continue
             pattern_parts.append(token)
    pattern = r'(' + '|'.join(pattern_parts) + r')'
    found_tokens = re.findall(pattern, input_str)
    input_token_indices = [token_to_idx.get(token, UNK_IDX) for token in found_tokens]
    target_token_indices = [SOS_IDX] + [token_to_idx.get(token, UNK_IDX) for token in target_list if token != PAD_TOKEN]
    # 确保返回的是 Tensor
    return torch.tensor(input_token_indices, dtype=torch.long), torch.tensor(target_token_indices, dtype=torch.long)

tokenize = tokenize_v3_revised # 设置为默认

# --- Transformer Model Definition (来自用户提供) ---
class PositionalEncoding(nn.Module): # Same as before
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout); position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model); pe[:, 0, 0::2] = torch.sin(position * div_term); pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        if x.size(1) == self.pe.size(0): x = x + self.pe[:x.size(0)]
        else: x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

class Seq2SeqTransformer(nn.Module):
    # (模型定义保持不变)
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int, emb_size: int, nhead: int, src_vocab_size: int, tgt_vocab_size: int, dim_feedforward: int = 512, dropout: float = 0.1, max_seq_len: int = 500):
        super().__init__()
        self.transformer = Transformer(d_model=emb_size, nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout, batch_first=True) # 确保 batch_first is True
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout, max_len=max_seq_len)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, trg: torch.Tensor,
                src_padding_mask: torch.Tensor, # Shape: (N, S)
                tgt_padding_mask: torch.Tensor, # Shape: (N, T)
                memory_key_padding_mask: torch.Tensor, # Shape: (N, S)
                tgt_mask: torch.Tensor): # Shape: (T, T)
        src_emb = self.positional_encoding(self.src_tok_emb(src)) # Shape: (N, S, E)
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg)) # Shape: (N, T, E)
        outs = self.transformer(src_emb, tgt_emb, src_mask=None, tgt_mask=tgt_mask,
                                memory_mask=None, src_key_padding_mask=src_padding_mask,
                                tgt_key_padding_mask=tgt_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: torch.Tensor, src_padding_mask: torch.Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor,
               tgt_padding_mask: torch.Tensor, memory_key_padding_mask: torch.Tensor):
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        return self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                                        memory_mask=None,
                                        tgt_key_padding_mask=tgt_padding_mask,
                                        memory_key_padding_mask=memory_key_padding_mask)

# --- Beam Search Definition (虽然不用，但保留定义) ---
def beam_search_decode(model, src_tensor, src_padding_mask, beam_width, max_len_out, device):
    # (beam_search_decode 函数定义保持不变)
    model.eval()
    with torch.no_grad():
        memory = model.encode(src_tensor, src_padding_mask); memory_key_padding_mask = src_padding_mask
        initial_beam = (0.0, torch.LongTensor([[SOS_IDX]]).to(device), [0.0]); beams = [initial_beam]; completed_beams = []
        for _ in range(max_len_out - 1):
            possible_continuations = []; new_beams_temp = []
            for beam_idx, (current_score, current_seq, scores_list) in enumerate(beams):
                if current_seq[0, -1].item() == EOS_IDX:
                    final_score = current_score; heapq.heappush(completed_beams, (final_score, current_seq, scores_list))
                    while len(completed_beams) > beam_width: heapq.heappop(completed_beams)
                    continue
                tgt_tensor = current_seq; tgt_padding_mask = (tgt_tensor == PAD_IDX)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_tensor.size(1), device=device)
                decoder_output = model.decode(tgt_tensor, memory, tgt_mask, tgt_padding_mask, memory_key_padding_mask)
                logits = model.generator(decoder_output[:, -1, :]); log_probs = F.log_softmax(logits, dim=-1)
                top_log_probs, top_indices = torch.topk(log_probs.squeeze(0), beam_width, dim=-1)
                for i in range(beam_width):
                    next_token_idx = top_indices[i].item(); log_prob = top_log_probs[i].item()
                    new_score = current_score + log_prob; new_seq = torch.cat([current_seq, torch.LongTensor([[next_token_idx]]).to(device)], dim=1)
                    new_scores_list = scores_list + [log_prob]; heapq.heappush(possible_continuations, (-new_score, new_seq, new_scores_list))
            k = beam_width - len(completed_beams)
            while possible_continuations and len(new_beams_temp) < k : neg_score, seq, scores = heapq.heappop(possible_continuations); new_beams_temp.append((-neg_score, seq, scores))
            beams = new_beams_temp
            if not beams: break
        while beams: score, seq, scores_list = beams.pop(0); heapq.heappush(completed_beams, (score, seq, scores_list)); heapq.heappop(completed_beams)
        sorted_beams = sorted([(s, seq, sl) for s, seq, sl in completed_beams], key=lambda x: x[0], reverse=True)
        results = []
        for score, seq_tensor, scores_per_token in sorted_beams:
             sequence_indices = seq_tensor.squeeze(0).tolist()[1:]; token_scores = scores_per_token[1:]
             results.append((sequence_indices, score, token_scores))
        return results[:beam_width]

# --- 辅助函数 (UI 相关) ---
# (format_card_html 和 display_hand 定义移到这里，在使用它们之前)
def format_card_html(card_value, revealed, is_opponent, is_drawn_card_display=False):
    """生成单张卡牌的 HTML 表示。"""
    if card_value is None: return ""
    if not isinstance(card_value, str) or len(card_value) < 2:
        return f'<div class="card white-card" title="无效卡牌数据: {card_value}">ERR</div>'
    color_char = card_value[0]; number = card_value[1:]
    card_class = "black-card" if color_char == 'B' else "white-card"
    content = ""; title_text = card_value
    if is_drawn_card_display: content = number; revealed = True
    elif revealed: content = number
    elif is_opponent: content = "?"; card_class += " hidden-opponent-html"; title_text = f"{'黑色' if color_char == 'B' else '白色'}未知牌"
    else: content = number; card_class += " hidden-self"; title_text = f"你的隐藏牌: {card_value}"
    text_color = "#ffffff" if color_char == 'B' else "#333333"
    if not revealed and is_opponent: text_color = "#ffffff" if color_char == 'B' else "#333333"
    if not content: content = "?" if not revealed else number
    card_html = f'<div class="card {card_class}" style="color: {text_color};" title="{title_text}">{content}</div>'
    if not is_opponent and revealed and not is_drawn_card_display: return f'<div title="{title_text}">{card_html}<div class="revealed-caption">(已揭示)</div></div>'
    else: return card_html

def display_hand(label, player_id, hand, revealed_list, is_opponent_hand, current_player_id):
    """显示指定玩家的一手牌。"""
    st.subheader(label)
    if not hand: st.write("(空)"); return
    if len(hand) != len(revealed_list): st.error(f"内部错误：玩家 {player_id} 的手牌 ({len(hand)}) 和揭示列表 ({len(revealed_list)}) 长度不匹配！"); return
    cols = st.columns(len(hand))
    for i, card in enumerate(hand):
        if i >= len(revealed_list): st.warning(f"玩家 {player_id} 手牌索引 {i} 超出揭示列表范围。"); continue
        if not isinstance(card, str) or len(card) < 2:
             st.warning(f"玩家 {player_id} 位置 {i} 的卡牌数据无效: {card}")
             with cols[i]: st.markdown('<div class="card white-card" title="无效卡牌">ERR</div>', unsafe_allow_html=True)
             continue
        with cols[i]:
            revealed = revealed_list[i]
            if is_opponent_hand and not revealed:
                 color_char = card[0]; button_key = f"guess_{i}"; button_text = "B ?" if color_char == 'B' else "W ?"
                 button_help = f"猜测对手位置 {i} 的牌"; wrapper_class = "black-card-button-wrapper" if color_char == 'B' else "white-card-button-wrapper"
                 is_active_player_turn = (current_player_id == HUMAN_PLAYER_ID)
                 st.markdown(f'<div class="{wrapper_class}">', unsafe_allow_html=True)
                 if st.button(button_text, key=button_key, help=button_help, disabled=not is_active_player_turn):
                     if is_active_player_turn:
                         st.session_state.selected_guess_index = i; st.session_state.guess_color_widget_val = 'B'; st.session_state.guess_value_widget_val = '0'
                         st.rerun()
                 st.markdown('</div>', unsafe_allow_html=True)
            else: st.markdown(format_card_html(card, revealed, is_opponent_hand), unsafe_allow_html=True)

# --- 辅助函数 (模型和对手逻辑) ---
def construct_model_input_string(env, current_state):
    """
    根据当前游戏状态构建模型所需的输入字符串。
    包含解析历史记录以添加对手之前猜错的信息。
    """

    opponent_player_id = current_state.get('current_player')
    human_player_id = 1 - opponent_player_id

    # 1. 构建对手手牌部分 (your hand)
    opponent_hand = []
    if hasattr(env, 'hands') and opponent_player_id < len(env.hands) and isinstance(env.hands[opponent_player_id], list):
        opponent_hand = env.hands[opponent_player_id]
    drawn_card = current_state.get('drawn_card')
    full_opponent_hand = sorted(opponent_hand + ([drawn_card] if drawn_card else []), key=env.card_value)
    your_hand_str = "".join([f"[{card}]" for card in full_opponent_hand])

    # 2. 解析历史记录，获取人类玩家每个位置被猜错的记录
    history = env.get_history() if hasattr(env, 'get_history') else []
    incorrect_guesses_map = {} # {position_index: set_of_wrong_values}

    opponent_id_str_en = f"Player {opponent_player_id}"
    other_id_str_zh = f"玩家 {1-opponent_player_id}"
    opponent_id_str_zh = f"玩家 {opponent_player_id}"
    incorrect_keywords = ["incorrectly", "错误", "不正确", "Incorrect"] # 增加英文大写可能
    guess_keywords = ["guessed", "猜测位置"]
    place_keywords = ["placed", "放置了"]
    cob = False
    last_entry = None
    for entry in history:
        is_opponent_guess = cob or (opponent_id_str_en in entry or opponent_id_str_zh in entry)
        is_incorrect = any(kw in entry for kw in incorrect_keywords)
        is_guess_action = cob or (any(kw in entry for kw in guess_keywords))
        cob = (opponent_id_str_en in entry or opponent_id_str_zh in entry) and (any(kw in entry for kw in guess_keywords))
        is_place = any(kw in entry for kw in place_keywords)
        if is_place and other_id_str_zh in entry:
            match = re.search(r"(?:position|位置)\s+(\d+)\s*(?:is|放置了)\s*([BW][0-9]+|[BW]-)", entry)
            position = int(match.group(1))
            try :
                largest_pos = max(incorrect_guesses_map.keys())
                if position <= largest_pos: 
                    #从 largest_pos 开始，反向循环到position
                    for i in range(largest_pos, position-1, -1):
                        if i in incorrect_guesses_map:
                            incorrect_guesses_map[i+1] = incorrect_guesses_map[i]
                        else:
                            incorrect_guesses_map[i] = set()
                            incorrect_guesses_map[i+1] = set()
                    incorrect_guesses_map[position] = set()
            except: pass
            
        if is_opponent_guess and is_incorrect and is_guess_action:
            # 尝试从日志中提取位置和猜错的牌值
            position = -1
            guessed_value = None
            try:
                # 尝试使用正则表达式匹配 "position <pos> is <card>" 或 "位置 <pos> 是 <card>"
                # 这个正则表达式需要根据你的确切日志格式调整
                match = re.search(r"(?:position|位置)\s+(\d+)\s*(?:is|是)\s*([BW][0-9]+|[BW]-)", last_entry)
                position = int(match.group(1))
                guessed_card = match.group(2)
                guessed_value = guessed_card[1:] # 提取 '5' 或 '-'
                # print(f"DEBUG History Parse Success: pos={position}, val={guessed_value} from '{entry}'") # 调试
            
                # 如果成功提取，添加到 map 中
                if position != -1 and guessed_value is not None:
                    if position not in incorrect_guesses_map:
                        incorrect_guesses_map[position] = set()
                    incorrect_guesses_map[position].add(guessed_value)

            except Exception as e:
                 st.warning(f"解析历史条目时出错: '{entry}' -> {e}") # 在 UI 中显示警告

        last_entry = entry

    st.write(f"DEBUG: Incorrect Guesses Map: {incorrect_guesses_map}") # 调试输出

    # 3. 构建人类手牌部分 (opponent's hand)，包含猜错信息
    human_hand = []; human_revealed = []; opponent_hand_parts = []
    if hasattr(env, 'hands') and human_player_id < len(env.hands) and isinstance(env.hands[human_player_id], list): human_hand = env.hands[human_player_id]
    if hasattr(env, 'revealed') and human_player_id < len(env.revealed) and isinstance(env.revealed[human_player_id], list): human_revealed = env.revealed[human_player_id]

    if len(human_hand) == len(human_revealed):
        for i, card in enumerate(human_hand):
            if i < len(human_revealed):
                if human_revealed[i]:
                    opponent_hand_parts.append(f"[{card}]")
                else:
                    # 构建基础表示 [Color?
                    base_repr = f"[{card[0]}?"
                    incorrect_set = incorrect_guesses_map.get(i)
                    # 如果有猜错记录，添加 :!guess1 guess2 ...
                    if incorrect_set:
                        # 对猜错的值进行排序以保证顺序一致性
                        sorted_incorrect = sorted(list(incorrect_set))
                        base_repr += f":!{' '.join(sorted_incorrect)}"
                    # 结束括号
                    base_repr += "]"
                    opponent_hand_parts.append(base_repr)
            else:
                st.warning(f"构建输入字符串时，人类玩家揭示列表索引 {i} 超出范围。")
                opponent_hand_parts.append("[?]") # 添加占位符
    else:
        st.warning("构建输入字符串时，人类玩家手牌与揭示列表长度不匹配。")
        opponent_hand_parts = ["[?]" for _ in human_hand] # 使用占位符

    opponent_hand_str = "".join(opponent_hand_parts)

    # 4. 组合最终字符串
    input_str = f"your hand: {your_hand_str}; opponent's hand: {opponent_hand_str}"

    # 5. 追加最近的猜测结果记录  —— 只取最后 6 条防止 prompt 过长
    recent_history = env.get_history()[-6:] if hasattr(env, "get_history") else []
    # 做一个极简压缩：去掉中文/英文“Player x / 玩家 x”前缀，只保留动作结果
    def _simplify(h):
        h = re.sub(r"(Player \d+|玩家 \d+)", "", h)
        h = h.strip()
        return h
    history_str = " | ".join(_simplify(h) for h in recent_history)
    if history_str:
        input_str += f"; recent history: {history_str}"

    return input_str

def generate_answer(messages, model="gpt-4o-mini", temperature=0):
    """
    调用 OpenAI ChatCompletion (>=1.0 SDK 写法)
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message.content.strip()


LOG_FILENAMEw="davinci_game_logw.txt"
def run_opponent_turn(env, current_state, thinking_placeholder, model_components):
    """
    对手 (OPPONENT_PLAYER_ID) 的回合逻辑 —— **改为调用 LLM 决策**。
    """
    opponent_player_id = current_state.get('current_player')
    if opponent_player_id != OPPONENT_PLAYER_ID:
        thinking_placeholder.empty()
        return current_state, "错误：非对手回合调用了对手逻辑。"

    human_player_id = HUMAN_PLAYER_ID
    thinking_placeholder.info(f"对手 (玩家 {opponent_player_id}) 正在思考…")

    # ---------- 1. 构建可读状态字符串 ----------
    input_str = construct_model_input_string(env, current_state)

    # ---------- 2. 询问 LLM ----------
    ## --- LLM ---   系统 / 用户 messages
    system_msg = (
        "You are an expert AI player of the game Da Vinci Code. "
        "Choose ONE legal action for this turn and output ONLY a JSON object. "
        "Legal action types:\n"
        "  • place : if the drawn card can be placed.\n"
        "  • guess : you must give 'position' (int) and 'card' (e.g. 'B5','W-').\n"
        "Return format examples:\n"
        '{"action":"place"}   OR   {"action":"guess","position":2,"card":"B5"}'
    )

    system_msg = (
        "You are an expert AI player of the game Da Vinci Code. "
        "The FIRST action of every turn **MUST** be a 'guess' as long as there is at least "
        "one unrevealed opponent card and the action ('guess', pos, card) is legal. "
        "Choose exactly ONE legal action and output ONLY a JSON object.\n"
        "Legal action types:\n"
        "  • place : use ONLY when guessing is impossible (e.g. all cards revealed).\n"
        "  • guess : include 'position' (int) and 'card' (e.g. 'B5','W-').\n"
        "when guess,You need to guess a card that has NOT yet appeared."
        "If the previous guess was wrong, then you should refrain from making the same kind of guess again."
        "Return format examples:\n"
        '{"action":"place"}   OR   {"action":"guess","position":2,"card":"B5"}'
    )

    system_msg = (
        # -------- 游戏角色与硬性规则 --------
        "You are an expert AI player of the game **Da Vinci Code**.  "
        "The FIRST action of every turn **MUST** be a 'guess' whenever there is at least one unrevealed opponent card. "
        "Output exactly ONE legal action as a JSON object, nothing else.\n"
        "Legal actions:  "
        "• guess → require keys `action`=`\"guess\"`, `position` (int), `card` (e.g. \"B5\", \"W-\").  "
        "• place → only allowed when no guessing is possible.\n\n"

        # --------  输入字符串格式说明 --------
        "### INPUT FORMAT\n"
        "`your hand:`  lists **all of your own cards**, already sorted left→right by game order.\n"
        "`opponent's hand:`  lists the rival's row; each item is:\n"
        "  • `[B7]` / `[W3]`  → revealed exact card (B=black, W=white, number 0-11 or - wildcard).\n"
        "  • `[B?]` / `[W?]`  → color known, value unknown yet.\n"
        "  • `[W?:!1 3 4]`    → same as above but the number is **NOT** 1,3,4 (numbers after `!` are previous wrong guesses for that position).\n"
        "`recent history:`  shows the last few actions in plain language.\n\n"

        # --------  规则与推理提示 --------
        "### GAME FACTS\n"
        "• The deck contains exactly one of each combination: color {B,W} × numbers {0-11}. Total 24 unique cards.\n"
        "• A card listed anywhere (your hand or revealed opponent slots) is no longer available elsewhere.\n"
        "• Each player’s row is ordered: lower numbers to the left; if numbers equal, black counts as lower than white.\n"
        "  Therefore every unknown value must lie **between its revealed neighbors** in that order.\n"
        "• Use: remaining unused cards  ∩  neighbor range  ∩  not-in `!wrong_list`  to narrow possibilities.\n\n"
        "The number on the right must be larger or equal to that on the left. If the numbers are the same, the white card is on the right."

        # --------  决策策略指令 --------
        "### DECISION STRATEGY\n"
        "1. Compute, for every unrevealed opponent slot, the set of still-possible numbers.\n"
        "2. Pick the slot whose candidate set is the **smallest** (i.e., easiest to guess).  "
        "Break ties by choosing the leftmost such slot.\n"
        "3. Within that slot, guess the **median** candidate value (or the only one if set size = 1).\n"
        "4. Return your decision strictly as JSON, e.g.  "
        "`{\"action\":\"guess\",\"position\":3,\"card\":\"W5\"}`  or  `{\"action\":\"place\"}`."
    )

    user_msg = (
        f"Current state:\n{input_str}\n"
        f"Drawn card: {current_state.get('drawn_card')}\n"
        f"Your legal actions this turn: {current_state.get('legal_actions')}\n"
        "Now decide and output JSON:"
    )
    try:
        llm_reply = generate_answer(
            messages=[{"role": "system", "content": system_msg},
                      {"role": "user", "content": user_msg}],
            model="o3",
            temperature=1
        )
    except Exception as e:
        llm_reply = f'{{"error":"{e}"}}'   # 记录后续日志

    # ---------- 3. 记录日志 ----------
    ## --- LLM ---
    try:
        with open(LOG_FILENAMEw, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] OPPONENT TURN\n")
            f.write(f"INPUT : {input_str}\n")
            f.write(f"LLMRAW: {llm_reply}\n")
    except Exception:
        pass  # 写日志失败不影响游戏

    # ---------- 4. 解析 LLM 输出 ----------
    import re                             # ← 函数顶部已有就省略
    opponent_action, correct_guess, turn_message = None, False, ""

    # ---- 清洗 LLM 回复，去掉 ```json``` 等围栏 ----
    clean_reply = llm_reply.strip()
    if clean_reply.startswith("```"):                # 去掉开头 ```json 或 ```
        clean_reply = re.sub(r"^```[\w]*\s*", "", clean_reply, flags=re.S)
        if clean_reply.endswith("```"):              # 去掉结尾 ```
            clean_reply = clean_reply[:-3]
    clean_reply = clean_reply.strip()

    # 若仍不是以 { 开头，尝试从中提取第一个 {...}
    if not clean_reply.startswith("{"):
        m = re.search(r"\{.*\}", clean_reply, flags=re.S)
        if m:
            clean_reply = m.group(0)

    # ---- 正式解析 ----
    try:
        parsed = json.loads(clean_reply)

        act_type = parsed.get("action")
        if act_type == "guess":
            pos = int(parsed["position"])
            card = str(parsed["card"])
            opponent_action = ("guess", pos, card)
            turn_message = f"对手让 LLM 猜测位置 {pos} 是 {card}..."
            # 判断正确与否
            actual_card = env.hands[human_player_id][pos]
            correct_guess = (actual_card == card)
        elif act_type == "place":
            opponent_action = ("place",)
            turn_message = "对手让 LLM 选择放置抽到的牌…"
    except Exception as e:
        # JSON 解析失败或字段缺失
        turn_message = f"LLM 输出无法解析 ({e})，改用后备策略。"

    # ---------- 5. 若 LLM 行为非法则退回原逻辑 ----------
    legal_actions = current_state.get('legal_actions', [])
    if opponent_action not in legal_actions:
        drawn_card_exists = current_state.get('drawn_card') is not None
        if ("place",) in legal_actions and drawn_card_exists:
            opponent_action = ("place",)
            turn_message += " 改为放置。"
        else:
            # 随机猜
            unrevealed = [i for i, r in enumerate(env.revealed[human_player_id]) if not r]
            pos = random.choice(unrevealed)
            card = random.choice(CARD_TOKENS)
            opponent_action = ("guess", pos, card)
            turn_message += f" 改为随机猜 {pos}-{card}。"
            actual_card = env.hands[human_player_id][pos]
            correct_guess = (actual_card == card)

    # ---------- 6. 执行动作 ----------
    new_state = current_state
    try:
        next_state, reward, done, info = env.step(opponent_action)
        new_state = next_state
        if opponent_action[0] == "guess":
            turn_message += " **猜对!**" if correct_guess else " **猜错.**"
    except Exception as e:
        turn_message += f" 执行动作时出错: {e}"

    thinking_placeholder.empty()

    # ---------- 7. 更新日志中的最终动作结果 ----------
    ## --- LLM ---
    try:
        with open(LOG_FILENAMEw, "a", encoding="utf-8") as f:
            f.write(f"ACT   : {opponent_action}\n")
            f.write(f"ENDMSG: {turn_message}\n\n")
    except Exception:
        pass

    return new_state, turn_message



# --- Dummy Model Loading / Creation Placeholder ---
@st.cache_resource # Cache the model and related components
def load_opponent_model_and_tokenizer():
    """加载对手模型和分词器组件 (占位符)"""
    st.write("正在加载对手模型和分词器 (或创建占位符)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"使用设备: {device}")
    # 定义模型参数 (应与训练时一致)
    
    # 初始化模型结构
    EMB_SIZE = 256; NHEAD = 8; FFN_HID_DIM = 512
    NUM_ENCODER_LAYERS = 3; NUM_DECODER_LAYERS = 3
    DROPOUT = 0.1; 
    MAX_SEQ_LEN_POS_ENC = 500
    model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD,
                           VOCAB_SIZE, VOCAB_SIZE, FFN_HID_DIM, DROPOUT,
                           max_seq_len=MAX_SEQ_LEN_POS_ENC).to(device)
    # --- 在此加载训练好的模型权重 ---
    st.warning("注意：当前使用的是随机初始化的模型，对手行为将不准确。请在代码中取消注释并提供正确的模型路径以加载训练好的权重。")
    model_path = './model.pth' # <-- **修改为你的模型路径**
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        st.success(f"成功加载模型权重: {model_path}")
    except FileNotFoundError:
        st.error(f"错误: 找不到模型文件 '{model_path}'。")
    except Exception as e:
        st.error(f"加载模型权重时出错: {e}。")
    # --- 结束模型加载 ---
    model.eval()
    tokenizer_func = tokenize
    vocab_info = { "SOS_IDX": SOS_IDX, "PAD_IDX": PAD_IDX, "EOS_IDX": EOS_IDX,
                   "idx_to_token": idx_to_token, "token_to_idx": token_to_idx,
                   "CARD_TOKENS": CARD_TOKENS, "UNK_TOKEN": UNK_TOKEN, "VOCAB_SIZE": VOCAB_SIZE }
    st.write("模型和分词器占位符准备就绪。")
    return model, tokenizer_func, vocab_info, device


# --- Streamlit 应用主逻辑 ---
# (基本保持不变, 除了加载模型和调用 run_opponent_turn)

st.title("达芬奇密码 - Web UI 对战 (模型对手)")

# 加载模型组件
try: model, tokenizer, vocab_info, device = load_opponent_model_and_tokenizer()
except Exception as e: st.error(f"加载模型或分词器时发生严重错误: {e}"); st.stop()

# 初始化 Session State
if 'game_env' not in st.session_state:
    try:
        st.session_state.game_env = DaVinciCodeGameEnvironment(); st.session_state.game_state = st.session_state.game_env.reset()
        st.session_state.selected_guess_index = None; st.session_state.guess_color_widget_val = 'B'; st.session_state.guess_value_widget_val = '0'
        st.session_state.message = "游戏初始化成功！"
    except Exception as e: st.error(f"初始化游戏环境时出错: {e}"); st.stop()

# 获取环境和状态
if 'game_env' not in st.session_state or 'game_state' not in st.session_state: st.error("游戏状态丢失，请刷新页面重试。"); st.stop()
env = st.session_state.game_env; state = st.session_state.game_state

# 显示消息
if 'message' in st.session_state and st.session_state.message: st.info(st.session_state.message); st.session_state.message = ""

# 检查状态有效性
if not isinstance(state, dict) or 'game_over' not in state:
     st.error("无效的游戏状态，尝试重置..."); # ... [重置逻辑] ...
     try:
         st.session_state.game_env = DaVinciCodeGameEnvironment(); st.session_state.game_state = st.session_state.game_env.reset()
         st.session_state.selected_guess_index = None; st.session_state.guess_color_widget_val = 'B'; st.session_state.guess_value_widget_val = '0'
         st.session_state.message = "游戏已重置。"; state = st.session_state.game_state; st.rerun()
     except Exception as e: st.error(f"重置游戏时出错: {e}"); st.stop()

# 检查游戏结束
if state.get('game_over', False):
    st.header("游戏结束!"); winner = state.get('winner')
    if winner is not None: st.success(f"玩家 {winner} 获胜！ 🎉")
    else: st.warning("游戏结束 (平局或错误)。")
    player_id = state.get('current_player', HUMAN_PLAYER_ID); opponent_id = OPPONENT_PLAYER_ID if player_id == HUMAN_PLAYER_ID else HUMAN_PLAYER_ID
    if hasattr(env, 'hands') and hasattr(env, 'revealed') and len(env.hands) > max(player_id, opponent_id) and len(env.revealed) > max(player_id, opponent_id):
        display_hand(f"玩家 {OPPONENT_PLAYER_ID} 的最终手牌", OPPONENT_PLAYER_ID, env.hands[OPPONENT_PLAYER_ID], env.revealed[OPPONENT_PLAYER_ID], False, state.get('current_player'))
        display_hand(f"玩家 {HUMAN_PLAYER_ID} 的最终手牌", HUMAN_PLAYER_ID, env.hands[HUMAN_PLAYER_ID], env.revealed[HUMAN_PLAYER_ID], False, state.get('current_player'))
    else: st.warning("无法显示最终手牌状态。")
    # --- 日志记录 ---
    try:
        final_state_info = {};
        if hasattr(env, 'get_state_info'): final_state_info = env.get_state_info()
        else: final_state_info = { 'winner': winner, 'hands': getattr(env, 'hands', 'N/A'), 'revealed': getattr(env, 'revealed', 'N/A'), 'history': getattr(env, 'history', 'N/A') }
        with open(LOG_FILENAME, "a", encoding="utf-8") as f:
            f.write("="*50 + "\n"); f.write(f"游戏结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"); f.write(f"获胜者: 玩家 {final_state_info.get('winner', 'N/A')}\n")
            if 'hands' in final_state_info and 'revealed' in final_state_info and isinstance(final_state_info['hands'], list) and isinstance(final_state_info['revealed'], list) and len(final_state_info['hands']) == len(final_state_info['revealed']):
                 for p_id in range(len(final_state_info['hands'])):
                      if p_id < len(final_state_info['hands']) and p_id < len(final_state_info['revealed']):
                          hand_list = final_state_info['hands'][p_id]; revealed_list = final_state_info['revealed'][p_id]
                          if isinstance(hand_list, list) and isinstance(revealed_list, list) and len(hand_list) == len(revealed_list):
                              hand_str = ', '.join(map(str, hand_list)); revealed_str_list = [f"{hand_list[i]}({'已揭示' if revealed_list[i] else '隐藏'})" for i in range(len(hand_list))]
                              revealed_str = ', '.join(revealed_str_list); f.write(f"\n玩家 {p_id} 最终手牌: {hand_str}\n"); f.write(f"玩家 {p_id} 最终状态: {revealed_str}\n")
                          else: f.write(f"\n玩家 {p_id} 最终手牌/状态数据格式错误。\n")
                      else: f.write(f"\n无法获取玩家 {p_id} 的最终手牌/状态数据。\n")
            else: f.write("\n无法记录最终手牌和状态信息。\n")
            if 'history' in final_state_info and isinstance(final_state_info['history'], list): f.write("\n游戏完整历史记录 (原始):\n"); [f.write(f"- {entry}\n") for entry in final_state_info['history']]
            else: f.write("\n无法记录游戏历史。\n")
            f.write("="*50 + "\n\n")
    except Exception as e: st.error(f"写入游戏日志时出错: {e}")
    # --- 结束日志 ---
    st.write("--- 游戏历史记录 (UI 显示) ---")
    if hasattr(env, 'get_history'):
        history = env.get_history(); filtered_history = []
        opponent_id_for_history = OPPONENT_PLAYER_ID; human_player_id_for_history = HUMAN_PLAYER_ID
        for entry in history:
            processed_entry = entry; player_name = ""; is_opponent_entry = False; is_human_entry = False
            opponent_player_str_en = f"Player {opponent_id_for_history}"; opponent_player_str_zh = f"玩家 {opponent_id_for_history}"
            human_player_str_en = f"Player {human_player_id_for_history}"; human_player_str_zh = f"玩家 {human_player_id_for_history}"
            if opponent_player_str_en in entry or opponent_player_str_zh in entry: player_name = f"对手 (玩家 {opponent_id_for_history})"; is_opponent_entry = True
            elif human_player_str_en in entry or human_player_str_zh in entry: player_name = f"你 (玩家 {human_player_id_for_history})"; is_human_entry = True
            if ("draws" in entry or "抽到" in entry) and is_opponent_entry: processed_entry = f"{player_name} 抽了一张牌。"
            elif ("placed" in entry or "放置了" in entry) and ("at position" in entry or "在位置" in entry):
                 position = "?"; reveal_text = ""; parts = entry.replace('(',' ').replace(')',' ').split(" ")
                 try:
                     pos_keywords = ["position", "位置"]; pos_index = -1
                     for kw in pos_keywords:
                         indices = [i for i, part in enumerate(parts) if kw in part]
                         if indices:
                             potential_pos_idx = indices[-1] + 1
                             if potential_pos_idx < len(parts) and parts[potential_pos_idx].split('-')[0].isdigit(): pos_index = potential_pos_idx; break
                     if pos_index != -1: position = parts[pos_index].split('-')[0]
                     if "revealed" in entry or "已揭示" in entry: reveal_text = "(已揭示)"
                     elif "hidden" in entry or "隐藏" in entry: reveal_text = "(隐藏)"
                     if player_name: processed_entry = f"{player_name} 在位置 {position} 放置了一张牌 {reveal_text}。"
                     else: processed_entry = f"有玩家在位置 {position} 放置了一张牌 {reveal_text}。"
                 except Exception:
                      if player_name: processed_entry = f"{player_name} 放置了一张牌。"
                      else: processed_entry = "有玩家放置了一张牌。"
            filtered_history.append(processed_entry)
        with st.expander("查看过滤后的历史记录", expanded=False):
             for entry in filtered_history: st.text(f"- {entry}")
        st.caption("注：抽牌及放置的具体牌面信息已隐藏。")
    if st.button("开始新游戏"):
        try:
            st.session_state.game_env = DaVinciCodeGameEnvironment(); st.session_state.game_state = st.session_state.game_env.reset()
            st.session_state.selected_guess_index = None; st.session_state.guess_color_widget_val = 'B'; st.session_state.guess_value_widget_val = '0'
            st.session_state.message = "新游戏开始！"; st.rerun()
        except Exception as e: st.error(f"开始新游戏时出错: {e}")
    st.stop()

# --- 确定玩家 ID 和回合 ---
is_human_turn = (state.get('current_player') == HUMAN_PLAYER_ID)

# --- 对手回合逻辑 ---
if not is_human_turn and not state.get('game_over', False):
    thinking_placeholder = st.empty()
    try:
        model_components = (model, tokenizer, vocab_info, device)
        new_opponent_state, opponent_message = run_opponent_turn(env, state, thinking_placeholder, model_components)
        st.session_state.game_state = new_opponent_state; st.session_state.message = opponent_message; st.session_state.selected_guess_index = None
        st.rerun()
    except Exception as e: thinking_placeholder.empty(); st.error(f"处理对手回合时发生意外错误: {e}")

# --- 游戏界面布局 ---
st.markdown("---")
current_player_id = state.get('current_player')
turn_indicator_text = f"--- 当前回合: 玩家 {current_player_id} ({'你' if current_player_id == HUMAN_PLAYER_ID else '对手'}) ---"
st.subheader(turn_indicator_text)

st.header(f"对手 (玩家 {OPPONENT_PLAYER_ID}) 的手牌")
if hasattr(env, 'hands') and hasattr(env, 'revealed') and OPPONENT_PLAYER_ID < len(env.hands) and OPPONENT_PLAYER_ID < len(env.revealed) and isinstance(env.hands[OPPONENT_PLAYER_ID], list) and isinstance(env.revealed[OPPONENT_PLAYER_ID], list):
    display_hand(f"对手 (玩家 {OPPONENT_PLAYER_ID})", OPPONENT_PLAYER_ID, env.hands[OPPONENT_PLAYER_ID], env.revealed[OPPONENT_PLAYER_ID], True, current_player_id)
else: st.write("等待对手或状态初始化...")

st.markdown("---")
st.header(f"你的手牌 (玩家 {HUMAN_PLAYER_ID})")
if hasattr(env, 'hands') and hasattr(env, 'revealed') and HUMAN_PLAYER_ID < len(env.hands) and HUMAN_PLAYER_ID < len(env.revealed) and isinstance(env.hands[HUMAN_PLAYER_ID], list) and isinstance(env.revealed[HUMAN_PLAYER_ID], list):
     display_hand(f"你 (玩家 {HUMAN_PLAYER_ID})", HUMAN_PLAYER_ID, env.hands[HUMAN_PLAYER_ID], env.revealed[HUMAN_PLAYER_ID], False, current_player_id)
else: st.write("等待加入或状态初始化...")

st.markdown("---")
col_info, col_actions = st.columns([2, 1])
with col_info:
    st.subheader("游戏信息"); st.write(f"牌堆剩余: {state.get('deck_size', 'N/A')} 张")
    drawn_card = state.get('drawn_card'); current_player_for_drawn_card = state.get('current_player')
    if drawn_card is not None and current_player_for_drawn_card is not None:
        drawn_card_owner = "你" if current_player_for_drawn_card == HUMAN_PLAYER_ID else f"对手 (玩家 {OPPONENT_PLAYER_ID})"
        st.markdown(f"""<div class="drawn-card-display"><span style="font-size: 0.9em; margin-bottom: 5px;">{drawn_card_owner} 抽到的牌:</span>{format_card_html(drawn_card, True, False, is_drawn_card_display=True)}</div>""", unsafe_allow_html=True)
    elif state.get('deck_size', -1) == 0 and not state.get('game_over'): st.warning("牌堆已空！本回合不再抽牌。")
    if is_human_turn:
        if state.get('drawn_card'):
            if state.get('can_guess_again'): st.info("你上一次猜对了！你可以再次猜测，或者放置你抽到的牌（这张牌将保持隐藏）。")
            else: st.info("轮到你了。请猜测对手的牌，或者放置你抽到的牌（这张牌将被揭示）。")
        elif state.get('deck_size', -1) == 0: st.info("轮到你了，牌堆已空，请直接猜测对手的牌。")
    elif not is_human_turn and not state.get('game_over'): st.write("等待对手操作...")
    st.write("--- 最近历史 ---")
    history = env.get_history() if hasattr(env, 'get_history') else []; filtered_history = []
    opponent_id_for_history = OPPONENT_PLAYER_ID; human_player_id_for_history = HUMAN_PLAYER_ID
    for entry in history:
        processed_entry = entry; player_name = ""; is_opponent_entry = False; is_human_entry = False
        opponent_player_str_en = f"Player {opponent_id_for_history}"; opponent_player_str_zh = f"玩家 {opponent_id_for_history}"
        human_player_str_en = f"Player {human_player_id_for_history}"; human_player_str_zh = f"玩家 {human_player_id_for_history}"
        if opponent_player_str_en in entry or opponent_player_str_zh in entry: player_name = f"对手 (玩家 {opponent_id_for_history})"; is_opponent_entry = True
        elif human_player_str_en in entry or human_player_str_zh in entry: player_name = f"你 (玩家 {human_player_id_for_history})"; is_human_entry = True
        if ("draws" in entry or "抽到" in entry) and is_opponent_entry: processed_entry = f"{player_name} 抽了一张牌。"
        elif ("placed" in entry or "放置了" in entry) and ("at position" in entry or "在位置" in entry):
             position = "?"; reveal_text = ""; parts = entry.replace('(',' ').replace(')',' ').split(" ")
             try:
                 pos_keywords = ["position", "位置"]; pos_index = -1
                 for kw in pos_keywords:
                     indices = [i for i, part in enumerate(parts) if kw in part]
                     if indices:
                         potential_pos_idx = indices[-1] + 1
                         if potential_pos_idx < len(parts) and parts[potential_pos_idx].split('-')[0].isdigit(): pos_index = potential_pos_idx; break
                 if pos_index != -1: position = parts[pos_index].split('-')[0]
                 if "revealed" in entry or "已揭示" in entry: reveal_text = "(已揭示)"
                 elif "hidden" in entry or "隐藏" in entry: reveal_text = "(隐藏)"
                 if player_name: processed_entry = f"{player_name} 在位置 {position} 放置了一张牌 {reveal_text}。"
                 else: processed_entry = f"有玩家在位置 {position} 放置了一张牌 {reveal_text}。"
             except Exception:
                  if player_name: processed_entry = f"{player_name} 放置了一张牌。"
                  else: processed_entry = "有玩家放置了一张牌。"
        filtered_history.append(processed_entry)
    start_index = max(0, len(filtered_history) - 15)
    for entry in filtered_history[start_index:]: st.text(f"- {entry}")
    st.caption("注：抽牌及放置的具体牌面信息已隐藏。")

with col_actions:
    st.subheader("你的操作")
    can_perform_action = is_human_turn and not state.get('game_over')
    can_place_action = can_perform_action and state.get('drawn_card') is not None
    can_guess_action = can_perform_action
    if can_perform_action:
        legal_actions = state.get('legal_actions', [])
        place_button_disabled = not (('place',) in legal_actions and can_place_action)
        if st.button("放置抽到的牌", key="place_card", disabled=place_button_disabled, help="将抽到的牌放入你的手牌中"):
            if not place_button_disabled:
                action = ('place',);
                try: next_state, reward, done, info = env.step(action); st.session_state.game_state = next_state; st.session_state.selected_guess_index = None; st.session_state.message = f"你放置了牌。奖励: {reward:.1f}"; st.rerun()
                except Exception as e: st.error(f"执行放置动作时出错: {e}")
        if 'selected_guess_index' in st.session_state and st.session_state.selected_guess_index is not None:
            idx_to_guess = st.session_state.selected_guess_index
            if hasattr(env, 'hands') and OPPONENT_PLAYER_ID < len(env.hands) and idx_to_guess < len(env.hands[OPPONENT_PLAYER_ID]):
                with st.container():
                     st.markdown('<div class="guess-container">', unsafe_allow_html=True); st.markdown(f"**猜测对手位置 {idx_to_guess} 的牌:**"); col1, col2 = st.columns(2)
                     with col1:
                         if 'guess_color_widget_val' not in st.session_state: st.session_state.guess_color_widget_val = 'B'
                         guess_color = st.selectbox("颜色:", ('B', 'W'), index=('B','W').index(st.session_state.guess_color_widget_val), key='guess_color_widget'); st.session_state.guess_color_widget_val = guess_color
                     with col2:
                         value_options = [str(i) for i in range(12)] + ['-']
                         if 'guess_value_widget_val' not in st.session_state: st.session_state.guess_value_widget_val = '0'
                         guess_value = st.selectbox("数值/百搭:", value_options, index=value_options.index(st.session_state.guess_value_widget_val), key='guess_value_widget'); st.session_state.guess_value_widget_val = guess_value
                     guessed_card = f"{guess_color}{guess_value}"; submit_guess_disabled = not can_guess_action
                     if st.button(f"确定猜测 {guessed_card}", key="submit_guess", disabled=submit_guess_disabled):
                         if not submit_guess_disabled:
                             action = ('guess', idx_to_guess, guessed_card)
                             try:
                                 next_state, reward, done, info = env.step(action); st.session_state.game_state = next_state; st.session_state.selected_guess_index = None
                                 if reward > 0.05: st.session_state.message = f"猜对了！奖励: {reward:.1f}"
                                 else: st.session_state.message = f"猜错了或猜测无效。奖励: {reward:.1f}"
                                 if 'guess_color_widget_val' in st.session_state: del st.session_state.guess_color_widget_val
                                 if 'guess_value_widget_val' in st.session_state: del st.session_state.guess_value_widget_val
                                 st.rerun()
                             except Exception as e: st.error(f"执行猜测动作时出错: {e}"); st.session_state.selected_guess_index = None; st.rerun()
                     if st.button("取消猜测", key="cancel_guess"):
                         st.session_state.selected_guess_index = None
                         if 'guess_color_widget_val' in st.session_state: del st.session_state.guess_color_widget_val
                         if 'guess_value_widget_val' in st.session_state: del st.session_state.guess_value_widget_val
                         st.rerun()
                     st.markdown('</div>', unsafe_allow_html=True)
            else: st.session_state.selected_guess_index = None; st.warning("选择的猜测位置已失效，请重新选择。"); st.rerun()
    elif not is_human_turn: st.write("等待对手操作...")
    else: st.write("现在无法操作。")

# --- 页脚或调试信息 (可选) ---
# ...
