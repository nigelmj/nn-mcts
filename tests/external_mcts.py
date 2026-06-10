from subprocess import PIPE, Popen

from src.game import Game


class ExternalMCTS:
    def __init__(self, iterations: int, game: Game) -> None:
        self.game = game
        game_name = game.__class__.__name__
        self.agent_process = Popen(
            [
                "java",
                "-cp",
                "",
                "tests.java.agents.MCTSRunner",
                game_name,
                str(self.game.current_player),
                str(self.game.size1),
                str(self.game.size2),
                str(iterations),
            ],
            stdout=PIPE,
            stdin=PIPE,
            text=True,
        )

    def get_move(self, turn: int = -1) -> int:
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
        if "," not in response:
            print(turn)
            print(response)

        x, y = response.split(",")
        x, y = int(x), int(y)
        if (x, y) == (-1, -1):
            # print("hello, skip move played")
            return self.game.size2 * self.game.size1
        return x * self.game.size2 + y
