"""État s = φ(RAM) : petit tuple discret (brique 3)."""
from nes_py.wrappers import JoypadSpace

from env import ACTIONS

A_IDS = {i for i, boutons in enumerate(ACTIONS) if "A" in boutons}  # actions qui appuient sur A


def phi(env: JoypadSpace, a_prev: int) -> tuple[int, int, int, int, int]:
    """Construit s_t à partir de la RAM et de l'action précédente a_{t-1}.

    s = (case_x, au_sol, a_tenu, ennemi, vitesse)
    """
    ram = env.unwrapped.ram
    x = int(ram[0x6D]) * 256 + int(ram[0x86])       # position de Mario en pixels
    case_x = x // 16                                 # 1 case = 1 bloc de 16 px
    au_sol = int(ram[0x1D] == 0)                     # 0x1D vaut 0 quand Mario touche le sol
    a_tenu = int(a_prev in A_IDS)                    # A était appuyé au pas précédent
    vx = int(ram[0x57])                              # vitesse horizontale (octet signé)
    vx = 0 if vx > 127 else vx                       # vers la gauche : compté comme 0
    vitesse = 0 if vx < 16 else (1 if vx < 32 else 2)  # 0 lent, 1 marche, 2 course
    ennemi = 0                                       # 0 aucun, 1 proche, 2 moyen
    for i in range(5):                               # 5 emplacements d'ennemis en RAM
        if ram[0x0F + i]:                            # emplacement actif
            dx = int(ram[0x6E + i]) * 256 + int(ram[0x87 + i]) - x
            if -8 <= dx <= 32:
                ennemi = 1
            elif dx <= 80 and ennemi == 0:
                ennemi = 2
    return (case_x, au_sol, a_tenu, ennemi, vitesse)


if __name__ == "__main__":
    from env import make_env, step_k

    env = make_env()
    env.reset()
    a = 3                                            # droite + B (courir)
    term = trunc = False
    while not (term or trunc):
        r, term, trunc, info = step_k(env, a, k=4)
        print(info["x_pos"], phi(env, a))
    env.close()