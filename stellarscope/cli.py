# -*- coding: utf-8 -*-

BANNER = {
    'future': r"""
                    ┏━┓╺┳╸┏━╸╻  ╻  ┏━┓┏━┓┏━┓┏━╸┏━┓┏━┓┏━╸
                    ┗━┓ ┃ ┣╸ ┃  ┃  ┣━┫┣┳┛┗━┓┃  ┃ ┃┣━┛┣╸ 
                    ┗━┛ ╹ ┗━╸┗━╸┗━╸╹ ╹╹┗╸┗━┛┗━╸┗━┛╹  ┗━╸
   Single-cell Transposable Element Locus Level Analysis of scRNA Sequencing
""",
    'starwars': r"""
     _______.___________. _______  __       __          ___      .______      
    /       |           ||   ____||  |     |  |        /   \     |   _  \     
   |   (----`---|  |----`|  |__   |  |     |  |       /  ^  \    |  |_)  |    
    \   \       |  |     |   __|  |  |     |  |      /  /_\  \   |      /     
.----)   |      |  |     |  |____ |  `----.|  `----./  _____  \  |  |\  \----.
|_______/       |__|     |_______||_______||_______/__/     \__\ | _| `._____|
               _______.  ______   ______   .______    _______ 
              /       | /      | /  __  \  |   _  \  |   ____|
             |   (----`|  ,----'|  |  |  | |  |_)  | |  |__
              \   \    |  |     |  |  |  | |   ___/  |   __|
          .----)   |   |  `----.|  `--'  | |  |      |  |____
          |_______/     \______| \______/  | _|      |_______|
 --Single-cell Transposable Element Locus Level Analysis of scRNA Sequencing--
""",
    'emboss': r"""
                      ┏━┛━┏┛┏━┛┃  ┃  ┏━┃┏━┃┏━┛┏━┛┏━┃┏━┃┏━┛
                      ━━┃ ┃ ┏━┛┃  ┃  ┏━┃┏┏┛━━┃┃  ┃ ┃┏━┛┏━┛
                      ━━┛ ┛ ━━┛━━┛━━┛┛ ┛┛ ┛━━┛━━┛━━┛┛  ━━┛
    Single-cell Transposable Element Locus Level Analysis of scRNA Sequencing
""",
}

def colorize(mstr, ljust=80, col='\x1b[40m\x1b[37m', defcol='\x1b[0m'):
    lines = [col + l.ljust(ljust) + defcol for l in mstr.split('\n')]
    return '\n'.join(lines)

def borderize(mstr, ljust = 80):
    u2502 = '\u2502'  # │ light vertical
    u2500 = '\u2500'  # ─ light horizontal
    u256D = '\u256D'  # ╭ light arc down and right
    u2570 = '\u2570'  # ╰ light arc up and right
    u256E = '\u256E'  # ╮ light arc down and left
    u256F = '\u256F'  # ╯ light arc up and left

    blines = [bline for bline in mstr.split('\n') if bline.strip()]
    ljust = max(ljust, max(map(len, blines))+2)
    blines = [bline.ljust(ljust-2) for bline in blines]
    blines = [f'{u2502}{bline}{u2502}' for bline in blines]
    return '\n'.join(
        [f'{u256D}{u2500 * (ljust-2)}{u256E}'] +
        blines +
        [f'{u2570}{u2500 * (ljust - 2)}{u256F}']
    )

BANNER_FUTURE = colorize(
   borderize(BANNER['future']),
    col = '\x1b[38;5;189m\x1b[48;5;17m'
)
BANNER_STARWARS = colorize(
    borderize(BANNER['starwars']),
    col = '\x1b[38;5;220m\x1b[48;5;235m'
)
