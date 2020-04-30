from typing import List
import numpy as np
from game.game import Action, Player, ActionHistory, AbstractGame
import gym


class ChineseChess(AbstractGame):
    def __init__(self, discount):
        super().__init__(discount)
        self.env = gym.make('gym_chinese_chess:cchess-v0')
        self.done = False
        self.observations = []
        self.observations.append(self.__get_observation())

    @property
    def action_space_size(self) -> int:
        return self.env.action_space.n

    def __get_observation(self):
        if self.env.current_player == 0:
            return self.env.generate_observation()
        elif self.env.current_player == 1:
            return - self.env.generate_observation()[:,::-1,::-1]
        else:
            raise RuntimeError("error player {}".format(self.env.current_player))

    def __reverse_pos(self,position):
        y, x = divmod(position, 9)
        y = 9 - y
        x = 8 - x
        return y * 9 + x

    def reverse_action(self,action: Action):
        action_ind = action.index
        from_act, to_act = divmod(action_ind, 90)
        from_act = self.__reverse_pos(from_act)
        to_act = self.__reverse_pos(to_act)
        return Action(from_act * 90 + to_act)

    def step(self, action: Action) -> int:
        if action.index >= 90 * 90:
            raise RuntimeError("action index out of range {}".format(action.index))

        if self.env.current_player == 0:
            pass
        elif self.env.current_player == 1:
            action = self.reverse_action(action)
        else:
            raise RuntimeError("error player {}".format(self.env.current_player))

        observation, reward, done, info = self.env.step(action.index)

        self.observations.append(self.__get_observation())
        self.done = done

        if self.env.current_player == 0:
            reward = -reward
        return reward

    def action2str(self,action:Action):
        return self.env.action2move(action)

    def terminal(self) -> bool:
        return self.done

    def legal_actions(self) -> List[Action]:
        la = self.env.get_possible_actions()
        action_list = [Action(i) for i in la]
        if self.env.current_player == 0:
            pass
        elif self.env.current_player == 1:
            action_list = [self.reverse_action(i) for i in action_list]
        else:
            raise RuntimeError("error player {}".format(self.env.current_player))
        return action_list

    def make_observation(self, state_index: int) -> np.array:
        return self.observations[state_index]

    def make_observation_image(self) -> np.array:
        return self.env.generate_image(mode="absolute")

    def make_observation_str(self) -> np.array:
        return self.env.render(mode="absolute")
