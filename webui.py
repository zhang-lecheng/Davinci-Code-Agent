import streamlit as st

# 设置页面配置，必须在所有其他Streamlit命令之前调用
st.set_page_config(layout="wide", page_title="达芬奇密码")

# 实现webui
import copy
import os
import random # 导入 random 模块
import time   # 导入 time 模块

# 确保 DaVinciCodeGameEnvironment 类在 env.py 文件中
# *** 已根据用户确认修改导入语句 ***
try:
    from env import DaVinciCodeGameEnvironment # 从 env.py 导入
except ImportError:
    st.error("错误：无法找到环境文件 'env.py' 或 'DaVinciCodeGameEnvironment' 类。请确保文件存在且路径正确。")
    st.stop() # 停止执行

# --- 样式定义 ---
# 使用 st.markdown 注入 CSS 来美化卡牌和按钮
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

    /* Streamlit 按钮的基本形状和大小调整 */
    div[data-testid="stButton"] > button {
        border-radius: 8px !important; /* 圆角 */
        padding: 5px !important; /* 内边距 */
        margin: 5px !important; /* 外边距 */
        width: 70px !important; /* 宽度 */
        height: 95px !important; /* 高度 */
        display: inline-flex !important;
        justify-content: center !important;
        align-items: center !important;
        text-align: center !important;
        font-size: 1.6em !important; /* 字体大小 */
        font-weight: bold !important;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1) !important;
        line-height: 1.2 !important;
        transition: background-color 0.2s ease !important;
    }

    /* --- 新增：针对包裹层的样式 --- */
    .black-card-button-wrapper, .white-card-button-wrapper {
        display: inline-block; /* 使包裹层适应按钮大小 */
        margin: 0;
        padding: 0;
        line-height: 0; /* 可能有助于对齐 */
    }

    /* --- 新增：特定按钮颜色样式 --- */
    .black-card-button-wrapper button {
        background-color: #333333 !important; /* 黑色背景 */
        color: #ffffff !important;           /* 白色文字 */
        border: 2px solid #555555 !important; /* 深色边框 */
    }
    .white-card-button-wrapper button {
        background-color: #ffffff !important; /* 白色背景 */
        color: #333333 !important;           /* 黑色文字 */
        border: 2px solid #cccccc !important; /* 浅色边框 */
    }

    /* 鼠标悬停时改变按钮外观 (可选) */
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
</style>
""", unsafe_allow_html=True)

# --- 辅助函数 ---

def format_card_html(card_value, revealed, is_opponent):
    """生成单张卡牌的 HTML 表示 (用于显示非交互式卡牌)"""
    if card_value is None: return "" # 处理空值
    if not isinstance(card_value, str) or len(card_value) < 2:
        return f'<div class="card white-card">?</div>' # 显示错误状态

    color_char = card_value[0]
    number = card_value[1:]
    card_class = "black-card" if color_char == 'B' else "white-card"
    content = ""

    if revealed:
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

    return f'<div class="card {card_class}" style="color: {text_color};">{content}</div>'

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
                 # 对手隐藏的牌显示为带样式的可点击按钮
                 color_char = card[0]
                 button_key = f"guess_{i}"
                 button_text = "B ?" if color_char == 'B' else "W ?"
                 button_help = f"猜测位置 {i} 的牌"

                 # 使用 Markdown 包裹按钮以应用 CSS 类
                 wrapper_class = "black-card-button-wrapper" if color_char == 'B' else "white-card-button-wrapper"
                 # 检查当前是否轮到人类玩家操作，如果不是，则禁用按钮
                 is_human_turn_for_button = (st.session_state.game_state.get('current_player') == 0) # 假设人类是玩家0

                 st.markdown(f'<div class="{wrapper_class}">', unsafe_allow_html=True)
                 # 只有在轮到人类玩家时才启用按钮点击功能
                 if st.button(button_text, key=button_key, help=button_help, disabled=not is_human_turn_for_button):
                     if is_human_turn_for_button: # 双重检查，确保回调只在启用时触发
                         st.session_state.selected_guess_index = i
                         st.session_state.guess_color = 'B' # 默认猜测输入
                         st.session_state.guess_value = '0' # 默认猜测输入
                         st.rerun()
                     else:
                         st.warning("现在不是你的回合！") # 理论上不会触发，因为按钮已禁用
                 st.markdown('</div>', unsafe_allow_html=True)

            else:
                # 显示玩家自己的牌或对手已揭示的牌 (使用 HTML)
                st.markdown(format_card_html(card, revealed, is_opponent_hand), unsafe_allow_html=True)


# --- Streamlit 应用主逻辑 ---

st.title("达芬奇密码 - Web UI 对战")

# 初始化 Session State
if 'game_env' not in st.session_state:
    try:
        st.session_state.game_env = DaVinciCodeGameEnvironment()
        st.session_state.game_state = st.session_state.game_env.reset() # 初始化游戏并获取初始状态
        st.session_state.selected_guess_index = None # 当前选择要猜测的牌的索引
        st.session_state.guess_color = 'B' # 猜测的颜色
        st.session_state.guess_value = '0' # 猜测的数值/百搭
        st.session_state.message = "游戏初始化成功！" # 用于显示游戏消息
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
    # 使用 st.empty() 来显示临时消息，并在下次 rerun 时自动清除
    msg_placeholder = st.empty()
    msg_placeholder.info(st.session_state.message)
    st.session_state.message = "" # 清除消息状态

# 检查游戏状态是否有效
if not isinstance(state, dict) or 'game_over' not in state:
     st.error("无效的游戏状态，尝试重置...")
     try:
         st.session_state.game_env = DaVinciCodeGameEnvironment()
         st.session_state.game_state = st.session_state.game_env.reset()
         st.session_state.selected_guess_index = None
         st.session_state.message = "游戏已重置。"
         state = st.session_state.game_state # 获取新状态
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

    st.write("--- 游戏历史记录 ---")
    if hasattr(env, 'get_history'):
        st.json(env.get_history())
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
human_player_id = 0 # 固定人类玩家为 0
opponent_player_id = 1
is_human_turn = (state.get('current_player') == human_player_id)

# --- 对手回合逻辑 ---
if not is_human_turn and not state.get('game_over', False):
    # 使用 placeholder 显示思考信息，避免 rerun 时残留
    thinking_placeholder = st.empty()
    thinking_placeholder.info(f"对手 (玩家 {opponent_player_id}) 正在思考...")
    time.sleep(1.5) # 暂停 1.5 秒模拟思考

    opponent_action = None
    must_place = False
    correct_guess = False # 初始化猜测结果标志

    # 获取人类玩家的隐藏牌信息 (直接从 env 获取)
    human_hand = env.hands[human_player_id]
    human_revealed = env.revealed[human_player_id]
    hidden_indices = [i for i, r in enumerate(human_revealed) if not r]

    drawn_card_opponent = state.get('drawn_card') # 对手抽到的牌

    # 检查对手是否有牌可操作
    if drawn_card_opponent is None:
         st.warning("对手回合但没有抽到牌（可能牌堆已空或状态错误）。")
         # 这里可能需要更复杂的逻辑，例如检查游戏是否应该结束
         # 为简单起见，我们假设这种情况不应发生或会自动处理
    elif not hidden_indices:
        must_place = True
        opponent_action = ('place',)
        st.session_state.message = f"对手没有可猜的牌，选择放置 {drawn_card_opponent}。"
    else:
        # 决定猜测动作
        target_index = random.choice(hidden_indices)
        actual_card = human_hand[target_index]

        if random.random() < 1/3: # 1/3 概率猜对
            opponent_action = ('guess', target_index, actual_card)
            st.session_state.message = f"对手猜测你的位置 {target_index} 是 {actual_card}..."
            correct_guess = True
        else: # 2/3 概率猜错
            possible_cards = [f"B{i}" for i in range(12)] + [f"W{i}" for i in range(12)] + ["B-", "W-"]
            wrong_guess_card = actual_card
            # 确保生成的错误猜测与实际卡牌不同
            while wrong_guess_card == actual_card:
                 wrong_guess_card = random.choice(possible_cards)
            opponent_action = ('guess', target_index, wrong_guess_card)
            st.session_state.message = f"对手猜测你的位置 {target_index} 是 {wrong_guess_card}..."
            correct_guess = False

    # --- 执行对手的动作 ---
    try:
        if opponent_action:
            # 执行第一步 (猜测或放置)
            next_state, reward, done, info = env.step(opponent_action)
            st.session_state.game_state = next_state # 更新状态

            # 根据猜测结果更新消息
            if opponent_action[0] == 'guess':
                 if correct_guess:
                     st.session_state.message += " 猜对了！"
                     # 如果猜对了，并且游戏没有结束，则立即执行放置动作
                     if not done:
                         time.sleep(1.0) # 短暂暂停
                         thinking_placeholder.info("对手猜对后选择放置卡牌...") # 更新提示信息
                         time.sleep(1.0)
                         place_action = ('place',)
                         # 检查放置动作是否合法 (理论上应该是合法的)
                         if place_action in next_state.get('legal_actions', []):
                              next_state_after_place, reward_place, done_after_place, info_place = env.step(place_action)
                              st.session_state.game_state = next_state_after_place # 再次更新状态
                              st.session_state.message += " 对手放置了卡牌。"
                              done = done_after_place # 更新游戏结束状态
                         else:
                              st.warning("对手无法在猜对后放置卡牌（意外情况）。对手回合结束。")
                              # 即使不能放置，回合也应该结束，因为不允许第二次猜测
                 else: # 猜错了
                     st.session_state.message += " 猜错了。"
                     # env.step 内部已经处理了放置揭示牌并切换回合

            elif opponent_action[0] == 'place':
                 st.session_state.message += " 对手放置了卡牌。"
                 # env.step 内部已经处理了放置并切换回合

        else:
             # 如果因为某种原因没有生成动作 (例如抽牌为空但游戏未结束)
             st.error("对手未能决定动作。跳过回合。")
             # 可能需要手动切换玩家或处理结束条件
             # 为简单起见，这里仅记录错误，依赖下一次 rerun 修正状态

        st.session_state.selected_guess_index = None # 清除可能残留的人类选择
        thinking_placeholder.empty() # 清除“正在思考”的消息
        time.sleep(0.5) # 短暂暂停显示结果
        st.rerun() # 重新运行以刷新整个界面，进入人类玩家回合或结束游戏

    except Exception as e:
        thinking_placeholder.empty() # 清除消息
        st.error(f"执行对手 ({opponent_player_id}) 动作时出错: {e}")
        # 记录错误，可能需要手动刷新或重置游戏

# --- 游戏界面布局 (在对手回合之后渲染) ---
st.markdown("---") # 分隔线

st.header(f"对手 (玩家 {opponent_player_id}) 的手牌")
if hasattr(env, 'hands') and hasattr(env, 'revealed') and \
   opponent_player_id < len(env.hands) and opponent_player_id < len(env.revealed) and \
   isinstance(env.hands[opponent_player_id], list) and isinstance(env.revealed[opponent_player_id], list):
    display_hand(f"对手 (玩家 {opponent_player_id})", env.hands[opponent_player_id], env.revealed[opponent_player_id], True)
else:
    st.write("等待对手或状态初始化...")


st.markdown("---") # 分隔线

st.header(f"你的手牌 (玩家 {human_player_id})")
if hasattr(env, 'hands') and hasattr(env, 'revealed') and \
   human_player_id < len(env.hands) and human_player_id < len(env.revealed) and \
   isinstance(env.hands[human_player_id], list) and isinstance(env.revealed[human_player_id], list):
     display_hand(f"你 (玩家 {human_player_id})", env.hands[human_player_id], env.revealed[human_player_id], False)
else:
     st.write("等待加入或状态初始化...")

st.markdown("---") # 分隔线

# --- 游戏信息和控制 ---
col_info, col_actions = st.columns([2, 1]) # 信息区和操作区分开

with col_info:
    st.subheader("游戏信息")
    st.write(f"牌堆剩余: {state.get('deck_size', 'N/A')} 张")
    st.write(f"当前回合: 玩家 {state.get('current_player', 'N/A')}")
    # 仅在轮到人类玩家时显示抽到的牌和提示
    if is_human_turn and state.get('drawn_card'):
        st.write(f"你抽到的牌: **{state['drawn_card']}**")
        if state.get('can_guess_again'):
            st.info("你上一次猜对了！你可以再次猜测，或者放置你抽到的牌（这张牌将保持隐藏）。")
        else:
            st.info("轮到你了。请猜测对手的牌，或者放置你抽到的牌（这张牌将被揭示）。")
    elif not is_human_turn and not state.get('game_over'): # 如果是对手回合且游戏未结束
        st.write("等待对手操作...") # 这个消息可能被上面的 "正在思考" 覆盖

    # 显示最近历史记录
    st.write("--- 最近历史 ---")
    history = env.get_history() if hasattr(env, 'get_history') else []
    start_index = max(0, len(history) - 5) # 显示最近 5 条
    for entry in history[start_index:]:
        st.text(f"- {entry}")


with col_actions:
    st.subheader("你的操作")

    # 仅在轮到人类玩家时显示操作按钮
    if is_human_turn and state.get('drawn_card'):
        legal_actions = state.get('legal_actions', [])
        can_place = ('place',) in legal_actions

        # 放置按钮
        if st.button("放置抽到的牌", key="place_card", disabled=not can_place, help="将抽到的牌放入你的手牌中"):
            if can_place:
                action = ('place',)
                try:
                    next_state, reward, done, info = env.step(action)
                    st.session_state.game_state = next_state
                    st.session_state.selected_guess_index = None # 清除猜测状态
                    st.session_state.message = f"你放置了牌。奖励: {reward:.1f}"
                    st.rerun() # 重新运行以更新 UI
                except Exception as e:
                    st.error(f"执行放置动作时出错: {e}")
            else:
                st.warning("当前无法执行放置动作。")


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

                     if st.button(f"确定猜测 {guessed_card}", key="submit_guess"):
                         action = ('guess', idx_to_guess, guessed_card)
                         is_legal = False
                         current_legal_actions = state.get('legal_actions', [])
                         for legal_act in current_legal_actions:
                             if legal_act == action:
                                 is_legal = True
                                 break

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
                         else:
                             st.warning(f"猜测动作 {action} 当前不合法。请检查可用的猜测。")

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
        # 在对手回合，操作区显示等待信息
        st.write("等待对手操作...")
    else:
         # 游戏刚开始或出现错误，没有抽到牌
         st.write("等待游戏开始或处理中...")

# --- 页脚或调试信息 (可选) ---
# st.markdown("---")
# with st.expander("调试信息 (当前状态)"):
#     st.json(state)
# with st.expander("调试信息 (完整环境状态)"):
#     st.json(env.get_state_info())
