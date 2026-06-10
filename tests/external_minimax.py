from subprocess import PIPE, Popen
from typing import Tuple

from src.game import Game


class ExternalMiniMax:
    def __init__(self, search_depth: int, game: Game):
        self.game = game
        game_name = game.__class__.__name__
        self.agent_process = Popen(
            [
                "java",
                "-cp",
                "",
                "tests.java.agents.MiniMaxRunner",
                game_name,
                str(self.game.current_player),
                str(self.game.size1),
                str(self.game.size2),
                str(search_depth),
            ],
            stdout=PIPE,
            stdin=PIPE,
            text=True,
        )

    def get_move(self, turn: int = None) -> int:
        board_strings = []
        for row in self.game.state:
            row_string = ""
            for piece in row:
                if piece == 0:
                    t = "0"
                elif piece == 1:
                    t = "1"
                else:
                    t = "2"
                row_string += t
            board_strings.append(row_string)
        board_string = ",".join(board_strings)

        command = f"{board_string}"
        if turn is not None:
            command += f";{turn}"

        # send the command to the agent process and get response
        self.agent_process.stdin.write(command + "\n")
        self.agent_process.stdin.flush()

        response = self.agent_process.stdout.readline().rstrip()

        x, y = response.split(",")
        x, y = int(x), int(y)
        return x * self.game.size2 + y
