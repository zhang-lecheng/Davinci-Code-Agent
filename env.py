# 实现游戏环境逻辑
# 核心功能，实现一个环境 Class
# 每次接受一个玩家的动作，给出反馈
# 涵盖所有环境信息


import random
import copy
import os

class DaVinciCodeGameEnvironment:
    """
    表示达芬奇密码的游戏环境。

    该类管理游戏状态、规则以及基于玩家动作的状态转换。
    它旨在供智能体（agent）或模拟使用，遵循类似于常见强化学习环境的结构。

    属性:
        num_players (int): 玩家数量（当前固定为 2）。
        deck (list):牌堆中剩余卡牌的列表。
        hands (list[list[str]]): 手牌列表，其中每个手牌是卡牌的列表。
        revealed (list[list[bool]]): 与手牌对应的已揭示状态列表。
        history (list[str]): 主要游戏事件的日志。
        current_player (int): 当前回合玩家的 ID（0 或 1）。
        game_over (bool): 标记游戏是否结束的标志。
        winner (int | None): 获胜玩家的 ID，如果是平局或进行中则为 None。
        _drawn_card (str | None): 当前玩家在其回合开始时抽到的卡牌。
        _can_guess_again (bool): 标记玩家是否正确猜中并可以再次行动的标志。
    """
    def __init__(self, num_players=2):
        """
        初始化游戏环境。

        Args:
            num_players (int): 玩家数量（当前支持 2 个）。
                                玩家 0 和玩家 1。
        """
        if num_players != 2:
            raise ValueError("当前仅支持 2 个玩家。")
        self.num_players = num_players
        # 初始化将被重置的属性
        self.deck = []
        self.hands = []
        self.revealed = []
        self.history = []
        self.current_player = 0
        self._drawn_card = None
        self._can_guess_again = False
        self.game_over = False
        self.winner = None
        # 调用 reset 来设置第一个游戏
        # self.reset() # reset 通常在外部调用以获取第一个状态

    def _initialize_game(self):
        """设置游戏变量的初始状态。"""
        self.black_joker = random.uniform(-0.5, 12)
        self.white_joker = random.uniform(-0.5, 12)
        # 初始化牌堆：黑色卡牌 (0-11)，白色卡牌 (0-11)，以及百搭牌 (-)
        self.deck = [f"B{i}" for i in range(12)] + [f"W{i}" for i in range(12)] + ["B-", "W-"]
        random.shuffle(self.deck)

        # 玩家手牌和揭示状态（使用按 player_id 索引的列表）
        self.hands = [[] for _ in range(self.num_players)]
        self.revealed = [[] for _ in range(self.num_players)]

        # 游戏历史记录
        self.history = []

        # 内部状态变量
        self.current_player = random.randint(0, self.num_players - 1)
        self._drawn_card = None # 回合开始时抽到的牌
        self._can_guess_again = False # 玩家是否猜对并能再次行动的标志
        self.game_over = False
        self.winner = None

        # 发初始手牌
        try:
            self._deal_initial_cards()
        except RuntimeError as e:
            self.history.append(f"初始化错误: {e}")
            self.game_over = True # 无法开始游戏
            return # 停止初始化

        # 记录开始
        self.history.append("游戏开始。")
        self.history.append(f"玩家 {self.current_player} 先行。")

        # 为起始玩家抽取第一张牌
        self._drawn_card = self._draw_card_from_deck()
        if self._drawn_card is None:
             self.game_over = True # 如果牌堆一开始就空了，则无法开始游戏
             self.history.append("错误：发牌后牌堆立即为空。")
        else:
             self.history.append(f"玩家 {self.current_player} 抽取他们的第一张牌。")


    def card_value(self, card):
        """
        返回用于排序目的的卡牌数值。
        处理百搭牌以及相同数字时颜色的优先级。

        Args:
            card (str): 卡牌字符串（例如 'B5', 'W-', 'W10'）。

        Returns:
            float: 卡牌的可排序值。对于无效卡牌返回无穷大。
        """
        if not isinstance(card, str) or len(card) < 2:
            return float('inf') # 有效卡牌不应发生这种情况

        color = card[0]
        value = card[1:]

        if value == '-':
            # 百搭牌最小：B- < W-
            if color == 'B':
                return self.black_joker
            elif color == 'W':
                return self.white_joker
            #return -1.0 if color == 'B' else -0.5
        try:
            num = int(value)
            # 数字优先，颜色次之（黑色 < 白色）
            color_value = 0.0 if color == 'B' else 0.5
            return num + color_value
        except ValueError:
            return float('inf') # 无效卡牌格式

    def _deal_initial_cards(self):
        """给每个玩家发 4 张牌并排序。"""
        num_initial_cards = 4
        if len(self.deck) < self.num_players * num_initial_cards:
             raise RuntimeError("牌堆中的牌不足以分发初始手牌。")

        for _ in range(num_initial_cards):
            for player_id in range(self.num_players):
                card = self._draw_card_from_deck()
                # 由于上面的检查，此处无需检查 None
                self.hands[player_id].append(card)

        # 对手牌进行排序并初始化揭示状态
        for player_id in range(self.num_players):
            self.hands[player_id].sort(key=self.card_value)
            self.revealed[player_id] = [False] * len(self.hands[player_id])
        self.history.append("已分发初始手牌。")

    def _draw_card_from_deck(self):
        """从牌堆中抽取一张牌。如果牌堆为空则返回 None。"""
        if not self.deck:
            return None
        return self.deck.pop()

    def insert_card(self, player_id, card, reveal=False):
        """
        将一张牌按正确的排序位置插入玩家手牌，
        并相应地更新揭示状态列表。

        Args:
            player_id (int): 玩家的 ID（0 或 1）。
            card (str): 要插入的卡牌。
            reveal (bool): 插入的卡牌是否应被揭示。

        Returns:
            int: 卡牌被插入的索引位置。
        """
        if player_id < 0 or player_id >= self.num_players:
             raise IndexError("无效的 player_id")
        if not card:
             raise ValueError("不能插入空卡牌")

        hand = self.hands[player_id]
        revealed_list = self.revealed[player_id]

        # 根据卡牌值查找正确的插入索引
        insert_pos = 0
        card_val = self.card_value(card)
        while insert_pos < len(hand) and self.card_value(hand[insert_pos]) < card_val:
            insert_pos += 1

        # 处理相同数字但颜色不同的卡牌（例如，当 W5 存在时插入 B5）
        while insert_pos < len(hand) and self.card_value(hand[insert_pos]) == card_val:
             # 确保相同数字时 B 在 W 之前
             if card[0] == 'B' and hand[insert_pos][0] == 'W':
                 break # 在白色之前插入黑色
             insert_pos += 1


        # 插入卡牌及其揭示状态
        hand.insert(insert_pos, card)
        revealed_list.insert(insert_pos, reveal)

        return insert_pos

    def reset(self):
        """
        将环境重置到新游戏的开始状态。

        Returns:
            dict: 从 _get_state() 获取的新游戏的初始状态。
        """
        self._initialize_game()
        # 初始化后第一个状态就绪（包括抽取第一张牌）
        return self._get_state()

    def _get_state(self):
        """
        构建当前玩家的状态表示。

        Returns:
            dict: 一个描述当前玩家视角下游戏状态的字典。包括：
                    'current_player', 'hand', 'revealed_self',
                    'opponent_hand_visible', 'opponent_revealed', 'deck_size',
                    'drawn_card', 'can_guess_again', 'legal_actions',
                    'game_over', 'winner'。
        """
        if self.game_over:
             # 返回终止状态表示
             player_id = self.current_player # 或者可能无关紧要
             opponent_id = (player_id + 1) % self.num_players
             return {
                'current_player': player_id,
                'hand': copy.deepcopy(self.hands[player_id]),
                'revealed_self': copy.deepcopy(self.revealed[player_id]),
                'opponent_hand_visible': [f"[{c}]" if r else f"[{c[0]}?]" for c, r in zip(self.hands[opponent_id], self.revealed[opponent_id])],
                'opponent_revealed': copy.deepcopy(self.revealed[opponent_id]),
                'deck_size': len(self.deck),
                'drawn_card': None,
                'can_guess_again': False,
                'legal_actions': [], # 没有可行动作
                'game_over': True,
                'winner': self.winner,
             }


        player_id = self.current_player
        opponent_id = (player_id + 1) % self.num_players # 假设有 2 个玩家

        # 创建对手可见手牌的表示
        opponent_hand_visible = []
        if opponent_id < len(self.hands): # 检查对手是否存在
            for i, card in enumerate(self.hands[opponent_id]):
                if self.revealed[opponent_id][i]:
                    opponent_hand_visible.append(card) # 完全显示已揭示的牌
                else:
                    # 对隐藏的牌，显示颜色，隐藏数值/百搭牌符号
                    opponent_hand_visible.append(f"{card[0]}?")

        opponent_revealed_status = copy.deepcopy(self.revealed[opponent_id]) if opponent_id < len(self.revealed) else []


        state = {
            'current_player': player_id,
            'hand': copy.deepcopy(self.hands[player_id]),
            'revealed_self': copy.deepcopy(self.revealed[player_id]),
            'opponent_hand_visible': opponent_hand_visible,
            'opponent_revealed': opponent_revealed_status, # 原始揭示状态列表
            'deck_size': len(self.deck),
            'drawn_card': self._drawn_card, # 玩家需要操作的牌
            'can_guess_again': self._can_guess_again, # 如果为 true，玩家可以猜测或放置（隐藏）
            'legal_actions': self._get_legal_actions(), # 根据当前状态计算合法动作
            'game_over': self.game_over,
            'winner': self.winner,
        }
        return state

    def _get_legal_actions(self):
        """
        根据状态确定当前玩家的合法动作。

        Returns:
            list: 有效动作元组的列表。
                  示例：[('place',), ('guess', 0, 'B5'), ('guess', 1, 'W-'), ...]
                  如果游戏结束或未抽牌，则返回空列表。
        """
        if self.game_over or self._drawn_card is None:
            # 如果玩家必须行动但没有抽到的牌（正常流程中不应发生），
            # 强制结束？或返回空列表。
            return []

        actions = []
        # 当持有牌时（_drawn_card 不为 None），'place' 始终是一个选项
        actions.append(('place',))

        # 如果对手有隐藏的牌，则可以进行猜测
        opponent_id = (self.current_player + 1) % self.num_players
        opponent_revealed = self.revealed[opponent_id]

        hidden_card_indices = [i for i, revealed in enumerate(opponent_revealed) if not revealed]

        if hidden_card_indices:
            # 生成所有可能的猜测卡牌值
            possible_cards = [f"B{i}" for i in range(12)] + [f"W{i}" for i in range(12)] + ["B-", "W-"]
            # 创建 ('guess', position, card_value) 元组
            for pos in hidden_card_indices:
                 for card_guess in possible_cards:
                     # 未来优化：根据已知卡牌修剪不可能的猜测
                     actions.append(('guess', pos, card_guess))

        return actions


    def step(self, action):
        """
        执行玩家的动作并转换游戏状态。

        Args:
            action (tuple): 当前玩家采取的动作。
                            必须是 _get_legal_actions() 返回的动作之一。
                            示例：('place',), ('guess', position, card_value)

        Returns:
            tuple: 包含以下内容的元组：
                - next_state (dict): 动作之后的状态，从 *新* 当前玩家（如果猜测正确则为同一玩家）的角度来看。
                - reward (float): 采取动作的玩家获得的奖励。
                                  +1 表示获胜，-1 表示失败（对手获胜），
                                  +/- 0.1 表示猜测结果，0 表示放置。
                - done (bool): 此动作后游戏是否结束。
                - info (dict): 附加信息（当前为空）。
        """
        if self.game_over:
            # 如果游戏已经结束，返回终止状态
            return self._get_state(), 0.0, True, {}

        # 验证动作
        # 注意：如果合法动作列表很大，检查它可能会很慢。
        # 更快的检查可能会验证动作格式和位置边界。
        # if action not in self._get_legal_actions(): # 这个检查可能很慢
        #     self.history.append(f"玩家 {self.current_player} 尝试非法动作: {action}")
        #     # 处理非法动作：惩罚并结束回合？还是引发错误？
        #     # 为了稳健性，尝试继续但记录错误并可能进行惩罚。
        #     # 一个简单的惩罚：结束回合，给予负奖励。
        #     reward = -0.5 # 非法动作的惩罚
        #     self._drawn_card = None
        #     self._can_guess_again = False
        #     self.current_player = (self.current_player + 1) % self.num_players
        #     self._drawn_card = self._draw_card_from_deck() # 为下一位玩家抽牌
        #     if self._drawn_card is None and not self.game_over:
        #         self.game_over = True; self.history.append("非法动作后牌堆为空。")
        #     else: self.history.append(f"玩家 {self.current_player} 抽到 {self._drawn_card}。")
        #     return self._get_state(), reward, self.game_over, {}
        # --- 结束基本的非法动作处理 ---

        player_id = self.current_player
        opponent_id = (player_id + 1) % self.num_players
        # 确保我们有牌可以操作（如果游戏未结束，应始终为 true）
        if self._drawn_card is None:
             # 这表示内部状态错误
             self.history.append(f"错误：轮到玩家 {player_id} 但没有抽到的牌。")
             self.game_over = True # 因错误结束游戏
             return self._get_state(), -1.0, True, {"error": "内部状态错误：没有抽到的牌"}


        drawn_card = self._drawn_card
        reward = 0.0
        done = False
        info = {}
        next_player = player_id # 默认：如果猜测正确，回合继续

        action_type = action[0]

        # --- 处理猜测动作 ---
        if action_type == 'guess':
            if len(action) != 3:
                 return self._handle_step_error(f"无效的猜测动作格式: {action}", player_id)

            _, position, card_guess = action

            # 验证位置
            if position < 0 or position >= len(self.hands[opponent_id]) or self.revealed[opponent_id][position]:
                 self.history.append(f"玩家 {player_id} 猜测了无效/已揭示的位置 {position}。视为不正确。")
                 actual_card = "N/A" # 标记为无效猜测场景
                 is_correct = False
            else:
                actual_card = self.hands[opponent_id][position]
                is_correct = (card_guess == actual_card)
                self.history.append(f"玩家 {player_id} 猜测位置 {position} 是 {card_guess}。")

            if is_correct:
                reward = 0.1 # 正确猜测的小奖励
                self.revealed[opponent_id][position] = True
                self.history.append(f"正确！位置 {position} 的牌揭示为 {actual_card}。")
                self._can_guess_again = True # 玩家可以再次行动（猜测或隐藏放置）

                # 检查获胜条件
                if all(self.revealed[opponent_id]):
                    self.history.append(f"玩家 {player_id} 揭示了所有对手的牌！")
                    reward = 1.0 # 获胜奖励
                    done = True
                    self.game_over = True
                    self.winner = player_id
                    self._drawn_card = None # 回合序列结束，不持有牌
                    self._can_guess_again = False
                    # 玩家此时不切换，游戏在他们的回合结束
                else:
                    # 回合继续给同一玩家 (next_player = player_id)
                    # 玩家为下一个动作保留相同的 _drawn_card
                    pass

            else: # 猜测错误（或无效位置）
                reward = -0.1 # 猜测错误导致揭牌的小惩罚
                self.history.append(f"位置 {position} 猜测错误。玩家 {player_id} 必须放置 {drawn_card}。")
                # 放置抽到的牌，揭示
                try:
                    insert_pos = self.insert_card(player_id, drawn_card, reveal=True)
                    self.history.append(f"玩家 {player_id} 在位置 {insert_pos} 放置了 {drawn_card} (已揭示)。")
                except Exception as e:
                     return self._handle_step_error(f"猜测错误后插入卡牌时出错: {e}", player_id)


                # 结束回合：重置标志，切换玩家，为下一位玩家抽牌
                self._drawn_card = None
                self._can_guess_again = False
                next_player = opponent_id
                # 为新玩家抽牌
                self._drawn_card = self._draw_card_from_deck()
                if self._drawn_card is None and not self.game_over:
                    self.history.append("牌堆为空，游戏结束。")
                    self.game_over = True # 或宣布平局
                    # 如果牌堆为空，检查当前玩家是否在此放置 *之前* 获胜
                    # 这个逻辑可能需要针对边缘情况进行细化，例如抽到最后一张牌并猜错
                elif not self.game_over:
                    self.history.append(f"玩家 {next_player} 抽到 {self._drawn_card}。")


        # --- 处理放置动作 ---
        elif action_type == 'place':
            if len(action) != 1:
                 return self._handle_step_error(f"无效的放置动作格式: {action}", player_id)

            # 揭示状态取决于玩家是否刚刚猜对 (_can_guess_again)
            reveal_status = not self._can_guess_again
            try:
                insert_pos = self.insert_card(player_id, drawn_card, reveal=reveal_status)
                reveal_text = "(已揭示)" if reveal_status else "(隐藏)"
                self.history.append(f"玩家 {player_id} 在位置 {insert_pos} 放置了 {drawn_card} {reveal_text}。")
            except Exception as e:
                 return self._handle_step_error(f"放置动作插入卡牌时出错: {e}", player_id)

            reward = 0.0 # 放置的中性奖励

            # 结束回合：重置标志，切换玩家，为下一位玩家抽牌
            self._drawn_card = None
            self._can_guess_again = False
            next_player = opponent_id
            # 为新玩家抽牌
            self._drawn_card = self._draw_card_from_deck()
            if self._drawn_card is None and not self.game_over:
                 self.history.append("牌堆为空，游戏结束。")
                 self.game_over = True # 或宣布平局
            elif not self.game_over:
                 self.history.append(f"玩家 {next_player} 抽到 {self._drawn_card}。")

        # --- 处理无效动作类型 ---
        else:
             return self._handle_step_error(f"无效的动作类型: {action_type}", player_id)


        # --- 完成步骤 ---
        self.current_player = next_player # 更新当前玩家

        # 检查对手是否因为当前玩家放置了他们最后一张揭示的牌而获胜
        # 这个检查很复杂，可能需要根据确切的获胜条件进行细化
        # 例如，在错误猜测时揭示你的最后一张牌是否会让你立即输掉？假设是。
        if action_type == 'guess' and not is_correct and all(self.revealed[player_id]):
             self.history.append(f"玩家 {player_id} 在错误猜测时揭示了他们的最后一张牌。玩家 {opponent_id} 获胜！")
             reward = -1.0 # player_id 的失败奖励
             done = True
             self.game_over = True
             self.winner = opponent_id
             self._drawn_card = None # 游戏结束，清除卡牌

        # 更新游戏结束状态
        if done:
            self.game_over = True

        # 获取 *新* 当前玩家的状态
        next_state = self._get_state()

        # 如果游戏刚刚结束，确保奖励反映了胜/负
        if self.game_over:
             if self.winner == player_id: # 刚刚行动的玩家获胜
                 reward = max(reward, 1.0) # 确保设置了获胜奖励
             elif self.winner == opponent_id: # 对手获胜
                 reward = min(reward, -1.0) # 确保设置了失败奖励


        return next_state, reward, self.game_over, info

    def _handle_step_error(self, message, player_id):
        """在步骤执行期间处理错误的内部辅助函数。"""
        self.history.append(f"玩家 {player_id} 的步骤错误: {message}")
        self.game_over = True # 出错时结束游戏
        # 返回指示错误的状态，负奖励，done=True
        error_state = self._get_state() # 获取完全处理错误之前的当前状态
        error_state['game_over'] = True
        error_state['legal_actions'] = []
        return error_state, -1.0, True, {"error": message}


    def render(self, mode='human'):
        """
        将当前游戏状态显示到控制台。

        Args:
            mode (str): 渲染模式（支持 'human'）。
        """
        if mode == 'human':
            # 简单的清屏用于控制台显示
            os.system('cls' if os.name == 'nt' else 'clear')
            print("\n" + "="*60)
            print("        达芬奇密码游戏环境")
            print("="*60 + "\n")

            # 显示两个玩家的手牌
            for p_id in range(self.num_players):
                is_current = p_id == self.current_player and not self.game_over
                turn_indicator = ">>>" if is_current else "   "
                print(f"{turn_indicator} 玩家 {p_id} 的手牌 ({len(self.hands[p_id])} 张):")

                hand_display = []
                for i, card in enumerate(self.hands[p_id]):
                    revealed_status = self.revealed[p_id][i]
                    # 显示格式：[Card] 如果已揭示，<Card> 如果隐藏但已知（当前玩家），[C?] 如果是隐藏的对手牌
                    if revealed_status:
                         display_card = f"[{card}]"
                    elif p_id == self.current_player and not self.game_over: # 当前玩家自己的隐藏牌
                         display_card = f"<{card}>"
                    else: # 对手的隐藏牌或游戏结束视图
                         display_card = f"[{card[0]}?]"

                    hand_display.append(display_card)
                print("   " + " ".join(hand_display))
                print("-" * 40)


            print(f"\n牌堆大小: {len(self.deck)}")
            # 仅当抽到的牌与当前玩家的动作相关时才显示
            if not self.game_over and self._drawn_card is not None and self.current_player is not None:
                 print(f"玩家 {self.current_player} 持有抽到的牌: {self._drawn_card}")
                 if self._can_guess_again:
                     print(f"玩家 {self.current_player} 可以再次猜测或放置 {self._drawn_card} (隐藏)。")
                 else:
                      print(f"玩家 {self.current_player} 必须猜测或放置 {self._drawn_card} (揭示)。")


            print("\n" + "="*60)
            print("最近历史记录 (最后 10 条):")
            start = max(0, len(self.history) - 10)
            for entry in self.history[start:]:
                print(f"- {entry}")
            print("="*60 + "\n")

            if self.game_over:
                print("########## 游戏结束 ##########")
                if self.winner is not None:
                    print(f"获胜者: 玩家 {self.winner}")
                else:
                    # 添加更具体的结束条件（如果跟踪的话，例如牌堆为空，双方失败）
                    print("游戏结束 (平局或错误)。")
                print("#############################")

        else:
            # 其他渲染模式的占位符（例如图形化）
            print(f"渲染模式 '{mode}' 未实现。")

    def get_history(self):
        """
        返回完整的游戏历史日志。

        Returns:
            list[str]: 描述游戏事件的字符串列表的副本。
        """
        return copy.deepcopy(self.history)

    def get_state_info(self):
        """
        返回内部游戏状态的综合字典，
        不与特定玩家的视角绑定。用于调试。

        Returns:
            dict: 详细的内部游戏状态，包括牌堆、所有手牌、
                  揭示状态、当前玩家、内部标志等。
        """
        return {
            'deck': copy.deepcopy(self.deck),
            'hands': copy.deepcopy(self.hands),
            'revealed': copy.deepcopy(self.revealed),
            'current_player': self.current_player,
            'drawn_card_internal': self._drawn_card,
            'can_guess_again_internal': self._can_guess_again,
            'game_over': self.game_over,
            'winner': self.winner,
            'history': copy.deepcopy(self.history)
        }
    def get_all_possible_cards(self):
        """
        返回所有可能的卡牌列表。

        Returns:
            list: 所有可能的卡牌字符串列表。
        """
        return [f"B{i}" for i in range(12)] + [f"W{i}" for i in range(12)] + ["B-", "W-"]

# --- 示例用法 ---
if __name__ == "__main__":
    env = DaVinciCodeGameEnvironment()
    state = env.reset() # 初始化并获取第一个状态
    done = False
    turn_count = 0
    max_turns = 100 # 安全限制

    print("开始达芬奇密码环境测试...")

    while not done and turn_count < max_turns:
        turn_count += 1
        env.render() # 显示状态
        print(f"\n--- 回合 {turn_count} ---")

        current_player_id = state['current_player']
        print(f"玩家 {current_player_id} 的回合。")
        print(f"手牌: {state['hand']}")
        print(f"对手可见牌: {state['opponent_hand_visible']}")
        print(f"需要打出的已抽牌: {state['drawn_card']}")
        print(f"能否再次猜测: {state['can_guess_again']}")

        legal_actions = state['legal_actions']
        print(f"合法动作数量: {len(legal_actions)}")

        if not legal_actions:
            print("没有可用的合法动作！结束游戏。")
            break # 如果没有可行动作则退出

        # --- 简单的随机智能体逻辑 ---
        # 真实的智能体（如 LLM 或 RL 智能体）会分析状态
        # 并在此处选择一个可能更好的动作。
        chosen_action = random.choice(legal_actions)
        # 可选：如果猜测选项很多，优先放置以加速测试
        # if len(legal_actions) > 10 and ('place',) in legal_actions:
        #      if random.random() < 0.7: # 如果可用，有 70% 的几率放置
        #           chosen_action = ('place',)

        print(f"玩家 {current_player_id} 随机选择了动作: {chosen_action}")
        # --- 结束智能体逻辑 ---

        # 在环境中执行选择的动作
        try:
            next_state, reward, done, info = env.step(chosen_action)
        except Exception as e:
            print(f"\n!!!!!!\nenv.step() 期间发生运行时错误: {e}\n!!!!!!")
            env.render() # 显示出错时的状态
            print("出错时的游戏历史记录:")
            for h in env.get_history(): print(f"- {h}")
            break # 出错时停止模拟


        print(f"动作产生的奖励: {reward}")
        print(f"游戏结束: {done}")
        if info: print(f"信息: {info}")

        state = next_state # 更新状态以进行下一次循环迭代

        if not done:
             input("按 Enter键 继续下一回合...") # 暂停以供查看
        else:
             print("\n游戏已结束。")


    print("\n--- 最终状态 ---")
    env.render() # 显示最终棋盘状态

    print(f"\n游戏在 {turn_count} 回合后结束。")
    final_info = env.get_state_info()
    print(f"获胜者: 玩家 {final_info['winner']}" if final_info['winner'] is not None else "获胜者: 无 (平局/错误)")

    # 可选：在最后打印完整的历史记录
    # print("\n完整游戏历史记录:")
    # for line in env.get_history():
    #     print(line)
