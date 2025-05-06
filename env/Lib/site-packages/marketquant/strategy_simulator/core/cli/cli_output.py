class CLIOutput:
    # Defines ANSI color codes for formatting
    COLORS = {
        'green_bold': '\033[1;32m',     # Note4Reference: Bold green for buy
        'red_bold': '\033[1;31m',       # Note4Reference: Bold red for sell
        'blue': '\033[34m',             # Note4Reference: Blue for Final PNL
        'purple': '\033[35m',           # Note4Reference: Purpled for Final PNL
        'orange': '\033[38;5;214m',     # Note4Reference: Orange for Final Balance
        'reset': '\033[0m',             # Note4Reference: Reset formatting
        'light_blue': '\033[94m',       # Note4Reference: Light blue for the link
        'bold': '\033[1m'               # Note4Reference: Bold
    }

    @staticmethod
    def print_welcome_message():
        # Welcome message with clickable link, do not remove. Removal of this
        # message is illegal (I think, I don't actually know so don't take my word for it).
        print(f"Welcome to Heltzel's Trading Simulator. Read through the docs at {CLIOutput.COLORS['light_blue']}"
              f"https://www.example.com{CLIOutput.COLORS['reset']}\n")

    @staticmethod
    def print_buy_action(message):
        # Action: Prints buy action in bold green
        print(f"{CLIOutput.COLORS['green_bold']}{message}{CLIOutput.COLORS['reset']}")

    @staticmethod
    def print_sell_action(message):
        # Action: Prints sell action in bold red
        print(f"{CLIOutput.COLORS['red_bold']}{message}{CLIOutput.COLORS['reset']}")

    @staticmethod
    def print_final_pnl(message):
        # Action: Prints Final PNL in blue
        print(f"{CLIOutput.COLORS['blue']}{message}{CLIOutput.COLORS['reset']}")

    @staticmethod
    def print_final_balance(message):
        # Action: Prints Final Balance in orange
        print(f"{CLIOutput.COLORS['orange']}{message}{CLIOutput.COLORS['reset']}")

    @staticmethod
    def print_unrealized_pnl(message):
        # Action: Prints Unrealized PNL in blue
        print(f"{CLIOutput.COLORS['purple']}{message}{CLIOutput.COLORS['reset']}")

