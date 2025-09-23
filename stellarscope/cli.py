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
    rlines = [_ for _ in mstr.split('\n') if _.strip()]
    ljust = max(ljust, max(map(len,rlines))+2)
    rlines = [_.ljust(ljust-2) for _ in rlines]
    rlines = [f'\u2502{_}\u2502' for _ in rlines]
    return '\n'.join(
        [f'\u256D{"\u2500" * (ljust-2)}\u256E'] +
        rlines +
        [f'\u2570{"\u2500" * (ljust - 2)}\u256F']
    )

BANNER_FUTURE = colorize(
   borderize(BANNER['future']),
    col = '\x1b[38;5;189m\x1b[48;5;17m'
)
BANNER_STARWARS = colorize(
    borderize(BANNER['starwars']),
    col = '\x1b[38;5;220m\x1b[48;5;235m'
)
