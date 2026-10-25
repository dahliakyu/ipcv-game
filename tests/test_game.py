"""Colour Clash game logic (prototype game.py), in pixels."""

import numpy as np
import pytest

from ipcvgame.core.config import load_config
from ipcvgame.interaction.game import Game, PaintSplat, Shield
from ipcvgame.scene.draw import player_color


@pytest.fixture
def game() -> Game:
    return Game(load_config()["game"])


def test_team_colours_are_red_and_blue():
    b, g, r = player_color(1)
    assert r > b and r > g          # P1 red
    b, g, r = player_color(2)
    assert b > r and b > g          # P2 blue


def test_new_player_has_full_health(game):
    p = game.get_player(1)
    assert p.health == game.max_health == 100
    assert game.get_player(1) is p


def test_reset_starts_a_fresh_round(game):
    p2 = game.get_player(2)
    p2.x, p2.bounds = 900.0, np.array([800.0, 100, 1000, 600])
    p2.health = 0
    p2.paint_splats.append(PaintSplat(0.5, 0.5, owner_id=1))
    game.shields[2] = Shield(900.0, 400.0, -1, remaining=1.0)
    game.winner_id = 1

    game.reset()

    assert game.winner_id is None
    assert p2.health == 100 and p2.paint_splats == []
    assert game.projectiles == [] and game.beams == [] and game.shields == {}
    assert p2.x == 900.0  # positions are kept


def test_reset_one_player(game):
    p1, p2 = game.get_player(1), game.get_player(2)
    p1.health, p2.health = 40, 50
    game.shields[1] = Shield(100.0, 400.0, 1, remaining=1.0)
    game.reset(1)
    assert p1.health == 100 and 1 not in game.shields
    assert p2.health == 50
