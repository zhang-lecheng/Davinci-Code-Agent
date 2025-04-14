import streamlit as st

# 设置页面配置，必须在所有其他Streamlit命令之前调用
st.set_page_config(layout="wide", page_title="达芬奇密码")

# 实现webui
import copy
import os
import random # 导入 random 模块
import datetime # <--- 新增：导入 datetime 模块

# 确保 DaVinciCodeGameEnvironment 类在 env.py 文件中
try:
    from env import DaVinciCodeGameEnvironment # 从 env.py 导入
except ImportError:
    st.error("错误：无法找到环境文件 'env.py' 或 'DaVinciCodeGameEnvironment' 类。请确保文件存在且路径正确。")
    st.stop() # 停止执行

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

# --- 辅助函数 ---

def format_card_html(card_value, revealed, is_opponent, is_drawn_card_display=False):
    """生成单张卡牌的 HTML 表示。"""
    if card_value is None: return ""
    if not isinstance(card_value, str) or len(card_value) < 2:
        return f'<div class="card white-card">?</div>'

    color_char = card_value[0]
    number = card_value[1:]
    card_class = "black-card" if color_char == 'B' else "white-card"
    content = ""

    if is_drawn_card_display:
        content = number
        revealed = True
    elif revealed:
        content = number
    elif is_opponent:
        content = "?"
        card_class += " hidden-opponent-html"
    else:
        content = number
        card_class += " hidden-self"

    text_color = "#ffffff" if color_char == 'B' else "#333333"
    if not revealed and is_opponent:
        text_color = "#ffffff" if color_char == 'B' else "#333333"

    if not content: content = "?" if not revealed else number

    card_html = f'<div class="card {card_class}" style="color: {text_color};">{content}</div>'

    if not is_opponent and revealed and not is_drawn_card_display:
        return f'<div>{card_html}<div class="revealed-caption">(已揭示)</div></div>'
    else:
        return card_html

def display_hand(label, hand, revealed_list, is_opponent_hand):
    """显示一手牌，对手隐藏牌使用带样式的按钮"""
    st.subheader(label)
    if not hand:
        st.write("(空)")
        return

    if len(hand) != len(revealed_list):
        st.error(f"内部错误：手牌 ({len(hand)}) 和揭示列表 ({len(revealed_list)}) 长度不匹配！")
        return

    cols = st.columns(len(hand))
    for i, card in enumerate(hand):
        if i >= len(revealed_list):
             st.warning(f"索引 {i} 超出揭示列表范围。")
             continue
        if not isinstance(card, str) or len(card) < 2:
             st.warning(f"位置 {i} 的卡牌数据无效: {card}")
             with cols[i]:
                 st.markdown('<div class="card white-card">ERR</div>', unsafe_allow_html=True)
             continue

        with cols[i]:
            revealed = revealed_list[i]
            if is_opponent_hand and not revealed:
                 color_char = card[0]
                 button_key = f"guess_{i}"
                 button_text = "B ?" if color_char == 'B' else "W ?"
                 button_help = f"猜测位置 {i} 的牌"
                 wrapper_class = "black-card-button-wrapper" if color_char == 'B' else "white-card-button-wrapper"
                 is_human_turn_for_button = (st.session_state.game_state.get('current_player') == 0)

                 st.markdown(f'<div class="{wrapper_class}">', unsafe_allow_html=True)
                 if st.button(button_text, key=button_key, help=button_help, disabled=not is_human_turn_for_button):
                     if is_human_turn_for_button:
                         st.session_state.selected_guess_index = i
                         st.session_state.guess_color = 'B'
                         st.session_state.guess_value = '0'
                         st.rerun()
                 st.markdown('</div>', unsafe_allow_html=True)

            else:
                st.markdown(format_card_html(card, revealed, is_opponent_hand), unsafe_allow_html=True)


# --- Streamlit 应用主逻辑 ---

st.title("达芬奇密码 - Web UI 对战")

# 初始化 Session State
if 'game_env' not in st.session_state:
    try:
        st.session_state.game_env = DaVinciCodeGameEnvironment()
        st.session_state.game_state = st.session_state.game_env.reset()
        st.session_state.selected_guess_index = None
        st.session_state.guess_color = 'B'
        st.session_state.guess_value = '0'
        st.session_state.message = "游戏初始化成功！"
    except Exception as e:
        st.error(f"初始化游戏环境时出错: {e}")
        st.stop()


# 从 session state 获取环境和状态
if 'game_env' not in st.session_state or 'game_state' not in st.session_state:
    st.error("游戏状态丢失，请刷新页面重试。")
    st.stop()

env = st.session_state.game_env
state = st.session_state.game_state

# 显示游戏消息
if 'message' in st.session_state and st.session_state.message:
    msg_placeholder = st.empty()
    msg_placeholder.info(st.session_state.message)
    st.session_state.message = ""

# 检查游戏状态是否有效
if not isinstance(state, dict) or 'game_over' not in state:
     st.error("无效的游戏状态，尝试重置...")
     try:
         st.session_state.game_env = DaVinciCodeGameEnvironment()
         st.session_state.game_state = st.session_state.game_env.reset()
         st.session_state.selected_guess_index = None
         st.session_state.message = "游戏已重置。"
         state = st.session_state.game_state
         st.rerun()
     except Exception as e:
         st.error(f"重置游戏时出错: {e}")
         st.stop()


# 检查游戏是否结束
if state.get('game_over', False):
    st.header("游戏结束!")
    winner = state.get('winner')
    if winner is not None:
        st.success(f"玩家 {winner} 获胜！ 🎉")
    else:
        st.warning("游戏结束 (平局或错误)。")

    player_id = state.get('current_player', 0)
    opponent_id = 1 - player_id
    if hasattr(env, 'hands') and hasattr(env, 'revealed') and \
       len(env.hands) > max(player_id, opponent_id) and len(env.revealed) > max(player_id, opponent_id):
        display_hand(f"玩家 {opponent_id} 的最终手牌", env.hands[opponent_id], env.revealed[opponent_id], False)
        display_hand(f"玩家 {player_id} 的最终手牌", env.hands[player_id], env.revealed[player_id], False)
    else:
        st.warning("无法显示最终手牌状态。")

    # --- 新增：游戏结束时记录日志 ---
    try:
        log_filename = "davinci_game_log.txt"
        # 尝试获取完整的最终状态信息
        final_state_info = {}
        if hasattr(env, 'get_state_info'):
             final_state_info = env.get_state_info()
        else: # 如果没有 get_state_info 方法，就用当前 state 和 env 属性构建
             final_state_info = {
                 'winner': winner,
                 'hands': env.hands if hasattr(env, 'hands') else 'N/A',
                 'revealed': env.revealed if hasattr(env, 'revealed') else 'N/A',
                 'history': env.history if hasattr(env, 'history') else 'N/A'
             }


        with open(log_filename, "a", encoding="utf-8") as f:
            f.write("="*50 + "\n")
            f.write(f"游戏结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"获胜者: 玩家 {final_state_info.get('winner', 'N/A')}\n")

            # 记录最终手牌和状态
            if 'hands' in final_state_info and 'revealed' in final_state_info and \
               isinstance(final_state_info['hands'], list) and isinstance(final_state_info['revealed'], list) and \
               len(final_state_info['hands']) == len(final_state_info['revealed']):
                 for p_id in range(len(final_state_info['hands'])):
                      # 安全地访问手牌和揭示列表
                      if p_id < len(final_state_info['hands']) and p_id < len(final_state_info['revealed']):
                          hand_list = final_state_info['hands'][p_id]
                          revealed_list = final_state_info['revealed'][p_id]
                          if isinstance(hand_list, list) and isinstance(revealed_list, list) and len(hand_list) == len(revealed_list):
                              hand_str = ', '.join(hand_list)
                              revealed_str_list = []
                              for card_idx, rev_status in enumerate(revealed_list):
                                   card_val = hand_list[card_idx]
                                   revealed_str_list.append(f"{card_val}({'已揭示' if rev_status else '隐藏'})")
                              revealed_str = ', '.join(revealed_str_list)

                              f.write(f"\n玩家 {p_id} 最终手牌: {hand_str}\n")
                              f.write(f"玩家 {p_id} 最终状态: {revealed_str}\n")
                          else:
                               f.write(f"\n玩家 {p_id} 最终手牌/状态数据格式错误。\n")
                      else:
                           f.write(f"\n无法获取玩家 {p_id} 的最终手牌/状态数据。\n")
            else:
                 f.write("\n无法记录最终手牌和状态信息。\n")


            # 记录完整原始历史记录到文件
            if 'history' in final_state_info and isinstance(final_state_info['history'], list):
                 f.write("\n游戏完整历史记录 (原始):\n")
                 for entry in final_state_info['history']:
                      f.write(f"- {entry}\n")
            else:
                 f.write("\n无法记录游戏历史。\n")

            f.write("="*50 + "\n\n")
        # 可选：通知用户日志已保存
        # st.sidebar.success(f"游戏记录已追加到 {log_filename}")

    except Exception as e:
        st.error(f"写入游戏日志时出错: {e}")
    # --- 结束日志记录 ---


    st.write("--- 游戏历史记录 (UI 显示) ---") # UI中仍然显示过滤后的历史
    if hasattr(env, 'get_history'):
        history = env.get_history()
        filtered_history = []
        opponent_id_for_history = 1
        human_player_id_for_history = 0

        for entry in history:
            processed_entry = entry
            player_name = ""
            is_opponent_entry = False
            is_human_entry = False

            opponent_player_str_en = f"Player {opponent_id_for_history}"
            opponent_player_str_zh = f"玩家 {opponent_id_for_history}"
            human_player_str_en = f"Player {human_player_id_for_history}"
            human_player_str_zh = f"玩家 {human_player_id_for_history}"

            if opponent_player_str_en in entry or opponent_player_str_zh in entry:
                player_name = f"对手 (玩家 {opponent_id_for_history})"
                is_opponent_entry = True
            elif human_player_str_en in entry or human_player_str_zh in entry:
                player_name = f"你 (玩家 {human_player_id_for_history})"
                is_human_entry = True

            if ("draws" in entry or "抽到" in entry) and is_opponent_entry:
                 processed_entry = f"{player_name} 抽了一张牌。"
            elif ("placed" in entry or "放置了" in entry) and ("at position" in entry or "在位置" in entry):
                 position = "?"
                 reveal_text = ""
                 parts = entry.replace('(',' ').replace(')',' ').split(" ")
                 try:
                     pos_keywords = ["position", "位置"]
                     pos_index = -1
                     for kw in pos_keywords:
                         indices = [i for i, part in enumerate(parts) if kw in part]
                         if indices:
                             potential_pos_idx = indices[-1] + 1
                             if potential_pos_idx < len(parts) and parts[potential_pos_idx].split('-')[0].isdigit():
                                  pos_index = potential_pos_idx
                                  break
                     if pos_index != -1:
                          position = parts[pos_index].split('-')[0]

                     if "revealed" in entry or "已揭示" in entry:
                         reveal_text = "(已揭示)"
                     elif "hidden" in entry or "隐藏" in entry:
                         reveal_text = "(隐藏)"

                     if player_name:
                          processed_entry = f"{player_name} 在位置 {position} 放置了一张牌 {reveal_text}。"
                     else:
                          processed_entry = f"有玩家在位置 {position} 放置了一张牌 {reveal_text}。"

                 except Exception:
                      if player_name:
                          processed_entry = f"{player_name} 放置了一张牌。"
                      else:
                          processed_entry = "有玩家放置了一张牌。"

            filtered_history.append(processed_entry)

        st.json(filtered_history) # 显示过滤后的历史
        st.caption("注：抽牌及放置的具体牌面信息已隐藏。")

    if st.button("开始新游戏"):
        try:
            st.session_state.game_env = DaVinciCodeGameEnvironment()
            st.session_state.game_state = st.session_state.game_env.reset()
            st.session_state.selected_guess_index = None
            st.session_state.message = "新游戏开始！"
            st.rerun()
        except Exception as e:
            st.error(f"开始新游戏时出错: {e}")
    st.stop()

# --- 确定玩家 ID 和回合 ---
human_player_id = 0
opponent_player_id = 1
is_human_turn = (state.get('current_player') == human_player_id)

# --- 对手回合逻辑 (无 sleep) ---
if not is_human_turn and not state.get('game_over', False):
    thinking_placeholder = st.empty()
    thinking_placeholder.info(f"对手 (玩家 {opponent_player_id}) 正在操作...")

    opponent_action = None
    must_place = False
    correct_guess = False

    if not hasattr(env, 'hands') or len(env.hands) <= human_player_id:
         st.error("无法获取人类玩家手牌信息，对手回合跳过。")
    else:
        human_hand = env.hands[human_player_id]
        human_revealed = env.revealed[human_player_id]
        hidden_indices = [i for i, r in enumerate(human_revealed) if not r]
        drawn_card_opponent = state.get('drawn_card')

        if drawn_card_opponent is None:
             st.warning("对手回合但没有抽到牌（可能牌堆已空或状态错误）。")
             # 如果没有抽到牌，对手只能猜测（如果可能）
             if hidden_indices:
                  target_index = random.choice(hidden_indices)
                  actual_card = human_hand[target_index]
                  if random.random() < 1/3:
                      opponent_action = ('guess', target_index, actual_card)
                      st.session_state.message = f"对手猜测你的位置 {target_index} 是 {actual_card}..."
                      correct_guess = True
                  else:
                      possible_cards = [f"B{i}" for i in range(12)] + [f"W{i}" for i in range(12)] + ["B-", "W-"]
                      wrong_guess_card = actual_card
                      while wrong_guess_card == actual_card:
                           wrong_guess_card = random.choice(possible_cards)
                      opponent_action = ('guess', target_index, wrong_guess_card)
                      st.session_state.message = f"对手猜测你的位置 {target_index} 是 {wrong_guess_card}..."
                      correct_guess = False
             else:
                  # 没牌抽，也没牌猜，回合无法进行
                  st.error("对手回合无法进行任何操作（无抽牌且无牌可猜）。")
                  # 这里可能应该结束游戏或跳过回合，取决于规则
                  opponent_action = None # 确保没有动作执行

        elif not hidden_indices:
            must_place = True
            opponent_action = ('place',)
            st.session_state.message = f"对手没有可猜的牌，选择放置。"
        else:
            target_index = random.choice(hidden_indices)
            actual_card = human_hand[target_index]
            if random.random() < 1/3:
                opponent_action = ('guess', target_index, actual_card)
                st.session_state.message = f"对手猜测你的位置 {target_index} 是 {actual_card}..."
                correct_guess = True
            else:
                possible_cards = [f"B{i}" for i in range(12)] + [f"W{i}" for i in range(12)] + ["B-", "W-"]
                wrong_guess_card = actual_card
                while wrong_guess_card == actual_card:
                     wrong_guess_card = random.choice(possible_cards)
                opponent_action = ('guess', target_index, wrong_guess_card)
                st.session_state.message = f"对手猜测你的位置 {target_index} 是 {wrong_guess_card}..."
                correct_guess = False

        try:
            if opponent_action:
                next_state, reward, done, info = env.step(opponent_action)
                st.session_state.game_state = next_state

                if opponent_action[0] == 'guess':
                     if correct_guess:
                         st.session_state.message += " 猜对了！"
                         if not done:
                             thinking_placeholder.info("对手猜对后选择放置卡牌...")
                             place_action = ('place',)
                             # 检查放置是否仍然是合法动作（重要，因为状态已更新）
                             if place_action in next_state.get('legal_actions', []):
                                  next_state_after_place, reward_place, done_after_place, info_place = env.step(place_action)
                                  st.session_state.game_state = next_state_after_place
                                  st.session_state.message += " 对手放置了卡牌。"
                                  done = done_after_place
                             else:
                                  # 如果猜对后不能放置（例如没有抽到牌），则回合结束
                                  st.warning("对手猜对但无法放置卡牌（可能因为牌堆已空）。对手回合结束。")
                                  st.session_state.message += " 但无法放置卡牌。"
                     else:
                         st.session_state.message += " 猜错了。"

                elif opponent_action[0] == 'place':
                     # 初始消息已包含放置意图
                     pass

            else:
                 # 如果 opponent_action 为 None (例如没牌抽也没牌猜)
                 st.error("对手未能决定动作。跳过回合。")
                 # 手动切换玩家？或者依赖环境的错误处理？
                 # 为安全起见，如果环境没自动切换，我们手动切换（如果游戏没结束）
                 if not state.get('game_over'):
                      env.current_player = human_player_id # 强制切换回人类
                      st.session_state.game_state = env._get_state() # 更新状态
                      st.session_state.message = "对手无法行动，回合跳过。"


            st.session_state.selected_guess_index = None
            thinking_placeholder.empty()
            st.rerun()

        except Exception as e:
            thinking_placeholder.empty()
            st.error(f"执行对手 ({opponent_player_id}) 动作时出错: {e}")


# --- 游戏界面布局 ---
st.markdown("---")

st.header(f"对手 (玩家 {opponent_player_id}) 的手牌")
if hasattr(env, 'hands') and hasattr(env, 'revealed') and \
   opponent_player_id < len(env.hands) and opponent_player_id < len(env.revealed) and \
   isinstance(env.hands[opponent_player_id], list) and isinstance(env.revealed[opponent_player_id], list):
    display_hand(f"对手 (玩家 {opponent_player_id})", env.hands[opponent_player_id], env.revealed[opponent_player_id], True)
else:
    st.write("等待对手或状态初始化...")


st.markdown("---")

st.header(f"你的手牌 (玩家 {human_player_id})")
if hasattr(env, 'hands') and hasattr(env, 'revealed') and \
   human_player_id < len(env.hands) and human_player_id < len(env.revealed) and \
   isinstance(env.hands[human_player_id], list) and isinstance(env.revealed[human_player_id], list):
     display_hand(f"你 (玩家 {human_player_id})", env.hands[human_player_id], env.revealed[human_player_id], False)
else:
     st.write("等待加入或状态初始化...")

st.markdown("---")

# --- 游戏信息和控制 ---
col_info, col_actions = st.columns([2, 1])

with col_info:
    st.subheader("游戏信息")
    st.write(f"牌堆剩余: {state.get('deck_size', 'N/A')} 张")
    st.write(f"当前回合: 玩家 {state.get('current_player', 'N/A')}")

    # 显示抽到的牌
    drawn_card = state.get('drawn_card')
    current_player_for_drawn_card = state.get('current_player')
    if drawn_card is not None and current_player_for_drawn_card is not None:
        drawn_card_owner = "你" if current_player_for_drawn_card == human_player_id else f"对手 (玩家 {opponent_player_id})"
        st.markdown(f"""
        <div class="drawn-card-display">
            <span style="font-size: 0.9em; margin-bottom: 5px;">{drawn_card_owner} 抽到的牌:</span>
            {format_card_html(drawn_card, True, False, is_drawn_card_display=True)}
        </div>
        """, unsafe_allow_html=True)
    # --- 新增：如果牌堆为空，明确提示 ---
    elif state.get('deck_size', -1) == 0 and not state.get('game_over'):
         st.warning("牌堆已空！本回合不再抽牌。")
    # --- 结束新增 ---


    # 操作提示
    if is_human_turn: # 轮到人类时
        if state.get('drawn_card'): # 如果有抽到牌
            if state.get('can_guess_again'):
                st.info("你上一次猜对了！你可以再次猜测，或者放置你抽到的牌（这张牌将保持隐藏）。")
            else:
                st.info("轮到你了。请猜测对手的牌，或者放置你抽到的牌（这张牌将被揭示）。")
        elif state.get('deck_size', -1) == 0: # 如果没抽到牌且牌堆为空
             st.info("轮到你了，牌堆已空，请直接猜测对手的牌。")
        # 其他情况（例如游戏刚开始，还没轮到行动）不显示提示

    elif not is_human_turn and not state.get('game_over'):
        st.write("等待对手操作...")

    # 显示最近历史记录 (过滤后)
    st.write("--- 最近历史 ---")
    history = env.get_history() if hasattr(env, 'get_history') else []
    filtered_history = []
    opponent_id_for_history = 1
    human_player_id_for_history = 0

    for entry in history:
        processed_entry = entry
        player_name = ""
        is_opponent_entry = False
        is_human_entry = False

        opponent_player_str_en = f"Player {opponent_id_for_history}"
        opponent_player_str_zh = f"玩家 {opponent_id_for_history}"
        human_player_str_en = f"Player {human_player_id_for_history}"
        human_player_str_zh = f"玩家 {human_player_id_for_history}"

        if opponent_player_str_en in entry or opponent_player_str_zh in entry:
            player_name = f"对手 (玩家 {opponent_id_for_history})"
            is_opponent_entry = True
        elif human_player_str_en in entry or human_player_str_zh in entry:
            player_name = f"你 (玩家 {human_player_id_for_history})"
            is_human_entry = True

        if ("draws" in entry or "抽到" in entry) and is_opponent_entry:
             processed_entry = f"{player_name} 抽了一张牌。"
        elif ("placed" in entry or "放置了" in entry) and ("at position" in entry or "在位置" in entry):
             position = "?"
             reveal_text = ""
             parts = entry.replace('(',' ').replace(')',' ').split(" ")
             try:
                 pos_keywords = ["position", "位置"]
                 pos_index = -1
                 for kw in pos_keywords:
                     indices = [i for i, part in enumerate(parts) if kw in part]
                     if indices:
                         potential_pos_idx = indices[-1] + 1
                         if potential_pos_idx < len(parts) and parts[potential_pos_idx].split('-')[0].isdigit():
                              pos_index = potential_pos_idx
                              break
                 if pos_index != -1:
                      position = parts[pos_index].split('-')[0]

                 if "revealed" in entry or "已揭示" in entry:
                     reveal_text = "(已揭示)"
                 elif "hidden" in entry or "隐藏" in entry:
                     reveal_text = "(隐藏)"

                 if player_name:
                      processed_entry = f"{player_name} 在位置 {position} 放置了一张牌 {reveal_text}。"
                 else:
                      processed_entry = f"有玩家在位置 {position} 放置了一张牌 {reveal_text}。"

             except Exception:
                  if player_name:
                      processed_entry = f"{player_name} 放置了一张牌。"
                  else:
                      processed_entry = "有玩家放置了一张牌。"

        filtered_history.append(processed_entry)

    start_index = max(0, len(filtered_history) - 5)
    for entry in filtered_history[start_index:]:
        st.text(f"- {entry}")
    st.caption("注：抽牌及放置的具体牌面信息已隐藏。")

with col_actions:
    st.subheader("你的操作")

    # 修改：当牌堆为空时，人类玩家不能放置，只能猜测
    can_perform_action = is_human_turn and not state.get('game_over')
    can_place_action = can_perform_action and state.get('drawn_card') is not None
    can_guess_action = can_perform_action # 只要轮到你且游戏没结束就可以尝试猜测

    if can_perform_action:
        legal_actions = state.get('legal_actions', [])
        # 放置按钮 (仅在有抽到牌时可用)
        place_button_disabled = not (('place',) in legal_actions and can_place_action)
        if st.button("放置抽到的牌", key="place_card", disabled=place_button_disabled, help="将抽到的牌放入你的手牌中"):
            if not place_button_disabled:
                action = ('place',)
                try:
                    next_state, reward, done, info = env.step(action)
                    st.session_state.game_state = next_state
                    st.session_state.selected_guess_index = None
                    st.session_state.message = f"你放置了牌。奖励: {reward:.1f}"
                    st.rerun()
                except Exception as e:
                    st.error(f"执行放置动作时出错: {e}")
            else:
                st.warning("当前无法执行放置动作（可能未抽牌或动作不合法）。")

        # 猜测输入框 (仅当点击了对手的牌时显示)
        if 'selected_guess_index' in st.session_state and st.session_state.selected_guess_index is not None:
            idx_to_guess = st.session_state.selected_guess_index
            if hasattr(env, 'hands') and opponent_player_id < len(env.hands) and idx_to_guess < len(env.hands[opponent_player_id]):
                with st.container():
                     st.markdown('<div class="guess-container">', unsafe_allow_html=True)
                     st.markdown(f"**猜测对手位置 {idx_to_guess} 的牌:**")
                     col1, col2 = st.columns(2)
                     with col1:
                         if 'guess_color_widget_val' not in st.session_state: st.session_state.guess_color_widget_val = 'B'
                         guess_color = st.selectbox("颜色:", ('B', 'W'), index=('B','W').index(st.session_state.guess_color_widget_val), key='guess_color_widget')
                         st.session_state.guess_color_widget_val = guess_color
                     with col2:
                         value_options = [str(i) for i in range(12)] + ['-']
                         if 'guess_value_widget_val' not in st.session_state: st.session_state.guess_value_widget_val = '0'
                         guess_value = st.selectbox("数值/百搭:", value_options, index=value_options.index(st.session_state.guess_value_widget_val), key='guess_value_widget')
                         st.session_state.guess_value_widget_val = guess_value

                     guessed_card = f"{guess_color}{guess_value}"

                     # 提交猜测按钮 (检查是否可以猜测)
                     submit_guess_disabled = not can_guess_action
                     if st.button(f"确定猜测 {guessed_card}", key="submit_guess", disabled=submit_guess_disabled):
                         if not submit_guess_disabled:
                             action = ('guess', idx_to_guess, guessed_card)
                             is_legal = False
                             current_legal_actions = state.get('legal_actions', [])
                             # 检查猜测动作是否在合法列表中 (注意：列表可能非常大)
                             # 简化检查：只要能猜测，就认为动作格式本身是合法的，让环境处理具体牌值
                             # if action in current_legal_actions: is_legal = True
                             # 假设只要能猜测，动作就是合法的，让环境判断对错
                             is_legal = True # 简化处理，依赖环境判断

                             if is_legal:
                                 try:
                                     next_state, reward, done, info = env.step(action)
                                     st.session_state.game_state = next_state
                                     st.session_state.selected_guess_index = None
                                     if reward > 0.05:
                                         st.session_state.message = f"猜对了！奖励: {reward:.1f}"
                                     else:
                                         st.session_state.message = f"猜错了。奖励: {reward:.1f}"
                                     if 'guess_color_widget_val' in st.session_state: del st.session_state.guess_color_widget_val
                                     if 'guess_value_widget_val' in st.session_state: del st.session_state.guess_value_widget_val
                                     st.rerun()
                                 except Exception as e:
                                     st.error(f"执行猜测动作时出错: {e}")
                                     st.session_state.selected_guess_index = None
                                     st.rerun()
                             # else: # 因为简化了合法性检查，这部分不再需要
                             #    st.warning(f"猜测动作 {action} 当前不合法。请检查可用的猜测。")

                     if st.button("取消猜测", key="cancel_guess"):
                         st.session_state.selected_guess_index = None
                         if 'guess_color_widget_val' in st.session_state: del st.session_state.guess_color_widget_val
                         if 'guess_value_widget_val' in st.session_state: del st.session_state.guess_value_widget_val
                         st.rerun()
                     st.markdown('</div>', unsafe_allow_html=True)
            else:
                 st.session_state.selected_guess_index = None
                 st.warning("选择的猜测位置已失效，请重新选择。")
                 st.rerun()

    elif not is_human_turn:
        st.write("等待对手操作...")
    else: # 轮到人类，但不能行动 (例如游戏结束或状态错误)
         st.write("现在无法操作。")


# --- 页脚或调试信息 (可选) ---
# st.markdown("---")
# with st.expander("调试信息 (当前状态)"):
#     st.json(state)
# with st.expander("调试信息 (完整环境状态)"):
#     st.json(env.get_state_info())

