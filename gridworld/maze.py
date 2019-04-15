# Copyright 2017 the pycolab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A scrolling maze to explore. Collect all of the coins!

Better Scrolly Maze is better than Scrolly Maze because it uses a much simpler
scrolling mechanism: cropping! As far as the pycolab engine is concerned, the
game world doesn't scroll at all: it just renders observations that are the size
of the entire map. Only later do "cropper" objects crop out a part of the
observation to give the impression of a moving world.

This cropping mechanism also makes it easier to derive multiple observations
from the same game, so the human user interface shows three views of the map at
once: a moving view that follows the player, another one that follows the
Patroller Sprite identified by the 'c' character, and a third that remains fixed
on a tantalisingly large hoard of gold coins, tempting the player to explore.

Regrettably, the cropper approach does mean that we have to give up the cool
starfield floating behind the map in Scrolly Maze. If you like the starfield a
lot, then Better Scrolly Maze isn't actually better.

Command-line usage: `better_scrolly_maze.py <level>`, where `<level>` is an
optional integer argument selecting Better Scrolly Maze levels 0, 1, or 2.

Keys: up, down, left, right - move. q - quit.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import curses
import gym
import sys
import numpy as np

from pycolab import ascii_art
from pycolab import cropping
from pycolab import human_ui
from pycolab import things as plab_things
from pycolab.prefab_parts import sprites as prefab_sprites


# pylint: disable=line-too-long
MAZES_ART = [
    # Each maze in MAZES_ART must have exactly one of the patroller sprites
    # 'a', 'b', and 'c'. I guess if you really don't want them in your maze, you
    # can always put them down in an unreachable part of the map or something.
    #
    # Make sure that the Player will have no way to "escape" the maze.
    #
    # Legend:
    #     '#': impassable walls.            'a': patroller A.
    #     '@': collectable coins.           'b': patroller B.
    #     'P': player starting location.    'c': patroller C.
    #     ' ': boring old maze floor.
    #
    # Finally, don't forget to update INITIAL_OFFSET and TEASER_CORNER if you
    # add or make substantial changes to a level.

    # Maze #1

    ['##############################',
     '# P @ @ @ @ @ @ @ @ @ @ @ @ @#',
     '#  @ @ @ @ @ @ @ @ @ @ @ @ @ #',
     '#######  a    ################',
     '# @ @ @ @ @ @ @ @ @ @ @ @ @ @#',
     '#  @ @ @ @ @ @ @ @ @ @ @ @ @ #',
     '###########    b  ############',
     '# @ @ @ @ @ @ @ @ @ @ @ @ @ @#',
     '#  @ @ @ @ @ @ @ @ @ @ @ @ @ #',
     '##############################'],
    
    # Maze #2
    ['###########   #################',
     '#         #   #               #',
     '#  ####   #####   #############',
     '#  #  #                       #',
     '#  #  #   ######  #######     #',
     '#  #  #####    #  #     #     #',
     '#  #           #  #     #     #',
     '#  #           #  #     #     #',
     '#P #           ####     #  @  #',
     '####                    #######']]


# The "teaser observations" (see docstring) have their top-left corners at these
# row, column maze locations. (The teaser window is 12 rows by 20 columns.)
TEASER_CORNER = [(0, 0),    # For level 0
                 (0, 0)]    # For level 1

# For dramatic effect, none of the levels start the game with the first
# observation centred on the player; instead, the view in the window is shifted
# such that the player is this many rows, columns away from the centre.
STARTER_OFFSET = [(0, 0),   # For level 0
                  (0, 0)]   # For level 1

# These colours are only for humans to see in the CursesUi.
COLOUR_FG = {' ': (0, 0, 0),        # Default black background
             '@': (999, 862, 110),  # Shimmering golden coins
             '#': (764, 0, 999),    # Walls of the maze
             'P': (0, 999, 999),    # This is you, the player
             'a': (999, 0, 780),    # Patroller A
             'b': (145, 987, 341)}  # Patroller B


COLOUR_BG = {'@': (0, 0, 0)}  # So the coins look like @ and not solid blocks.


def make_game(level):
  """Builds and returns a Better Scrolly Maze game for the selected level."""
  return ascii_art.ascii_art_to_game(
      MAZES_ART[level], what_lies_beneath=' ',
      sprites={
          'P': PlayerSprite,
          'a': PatrollerSprite,
          'b': PatrollerSprite},
      drapes={
          '@': CashDrape},
      update_schedule=['a', 'b', 'P', '@'],
      z_order='ab@P')


def make_croppers(level):
  """Builds and returns `ObservationCropper`s for the selected level.

  We make three croppers for each level: one centred on the player, one centred
  on one of the Patrollers (scary!), and one centred on a tantalising hoard of
  coins somewhere in the level (motivating!)

  Args:
    level: level to make `ObservationCropper`s for.

  Returns:
    a list of three `ObservationCropper`s.
  """
  return [
      # The player view.
      cropping.ScrollingCropper(rows=10, cols=30, to_track=['P'],
                                initial_offset=STARTER_OFFSET[level]),
      # The teaser!
      cropping.FixedCropper(top_left_corner=TEASER_CORNER[level],
                            rows=10, cols=31, pad_char=' '),
  ]


class PlayerSprite(prefab_sprites.MazeWalker):
  """A `Sprite` for our player, the maze explorer."""

  def __init__(self, corner, position, character):
    """Constructor: just tells `MazeWalker` we can't walk through walls."""
    super(PlayerSprite, self).__init__(
        corner, position, character, impassable='#')

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del backdrop, things, layers  # Unused

    if actions == 0:    # go upward?
      self._north(board, the_plot)
    elif actions == 1:  # go downward?
      self._south(board, the_plot)
    elif actions == 2:  # go leftward?
      self._west(board, the_plot)
    elif actions == 3:  # go rightward?
      self._east(board, the_plot)
    # elif actions == 4:  # stay put? (Not strictly necessary.)
    #   self._stay(board, the_plot)
    # if actions == 5:    # just quit?
    #   the_plot.terminate_episode()


class PatrollerSprite(prefab_sprites.MazeWalker):
  """Wanders back and forth horizontally, killing the player on contact."""

  def __init__(self, corner, position, character):
    """Constructor: list impassables, initialise direction."""
    super(PatrollerSprite, self).__init__(
        corner, position, character, impassable='#')
    # Choose our initial direction based on our character value.
    self._moving_east = bool(ord(character) % 2)

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del actions, backdrop  # Unused.

    # We only move once every two game iterations.
    if the_plot.frame % 2:
      self._stay(board, the_plot)  # Also not strictly necessary.
      return

    # If there is a wall next to us, we ought to switch direction.
    row, col = self.position
    if layers['#'][row, col-1]: self._moving_east = True
    if layers['#'][row, col+1]: self._moving_east = False

    # Make our move. If we're now in the same cell as the player, it's instant
    # game over!
    (self._east if self._moving_east else self._west)(board, the_plot)
    if self.position == things['P'].position: the_plot.terminate_episode()


class CashDrape(plab_things.Drape):
  """A `Drape` handling all of the coins.

  This Drape detects when a player traverses a coin, removing the coin and
  crediting the player for the collection. Terminates if all coins are gone.
  """

  def update(self, actions, board, layers, backdrop, things, the_plot):
    # If the player has reached a coin, credit one reward and remove the coin
    # from the scrolling pattern. If the player has obtained all coins, quit!
    player_pattern_position = things['P'].position

    if self.curtain[player_pattern_position]:
      the_plot.log('Coin collected at {}!'.format(player_pattern_position))
      the_plot.add_reward(100)
      self.curtain[player_pattern_position] = False
      if not self.curtain.any(): the_plot.terminate_episode()

class MazeEnv(gym.Env):
    """
    Wrapper to adapt to OpenAI's gym interface.
    didn't make any cropping yet 
    """
    action_space = gym.spaces.Discrete(4)  
    observation_space = gym.spaces.Box(low=0, high=1, shape=[10, 30, 5], dtype=np.uint8) # need to change row, column number when modify environment
    def _to_obs(self, observation):
        hallway = observation.layers[' '] 
        ob = np.stack([observation.layers[c] for c in 'Pab@'] + [hallway], axis=2).astype(np.uint8)
        return ob

    def reset(self):
        self._game = make_game(0)
        observation, reward, _ = self._game.its_showtime()
        return self._to_obs(observation)

    def step(self, action):
        observation, reward, _ = self._game.play(action)
        if reward is None: reward = 0
        done = self._game.game_over
        info = {}
        return self._to_obs(observation), reward, done, info
    
    # def render(self):
    #     croppers = make_croppers(0)
    #     ui = human_ui.CursesUi(
    #         keys_to_actions={curses.KEY_UP: 0, curses.KEY_DOWN: 1,
    #                         curses.KEY_LEFT: 2, curses.KEY_RIGHT: 3,
    #                         -1: 4,
    #                         'q': 5, 'Q': 5},
    #         delay=100, colour_fg=COLOUR_FG, colour_bg=COLOUR_BG,
    #         croppers=croppers)
    #     ui._display(self._game)




def main(argv=()):
  level = int(argv[1]) if len(argv) > 1 else 0

  # Build a Better Scrolly Maze game.
  game = make_game(level)
  # Build the croppers we'll use to scroll around in it, etc.
  croppers = make_croppers(level)

  # Make a CursesUi to play it with.
  ui = human_ui.CursesUi(
      keys_to_actions={curses.KEY_UP: 0, curses.KEY_DOWN: 1,
                       curses.KEY_LEFT: 2, curses.KEY_RIGHT: 3,
                       -1: 4,
                       'q': 5, 'Q': 5},
      delay=100, colour_fg=COLOUR_FG, colour_bg=COLOUR_BG,
      croppers=croppers)

  # Let the game begin!
  ui.play(game)


if __name__ == '__main__':
  main(sys.argv)

