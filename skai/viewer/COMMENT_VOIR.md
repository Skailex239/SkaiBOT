# 👀 Comment regarder ton IA jouer

## En ce moment (aperçu live)
Un serveur tourne dans l'environnement : ouvre l'aperçu **« SkaiBOT visualiseur »** (port 8080).
C'est une vraie partie OpenFront qui se joue : l'IA verte conquiert du terrain face aux bots.

## Chez toi, plus tard
1. Installe le moteur une fois :
   ```bash
   cd base-game && npm install && cd ..
   ```
2. Lance le visualiseur (Python 3.11/3.12, aucune bibliothèque à installer) :
   ```bash
   python skai/viewer/server.py
   ```
3. Ouvre ton navigateur sur **http://localhost:8080**.

## Important : quelle "IA" tu regardes ?
Pour l'instant c'est une **politique d'expansion simple** (écrite à la main) : elle attaque le
terrain libre autour d'elle en gardant des troupes en réserve. Le but est de *voir une vraie
partie en marche* et de valider toute la chaîne (pont → partie → affichage).

La **vraie** IA (le réseau de neurones PPO qui apprend) remplacera cette politique dans
`agent_policy.py` après l'étape d'entraînement/self-play (voir `docs/ROADMAP.md`). Le visualiseur,
lui, ne changera pas : il affichera les décisions du modèle entraîné.

Prochaines étapes possibles :
- entraîner un premier modèle PPO et le brancher dans cette page ;
- ajouter le mode **0 nation / 400 tribus** et la récompense speedrun ;
- développer le **self-play** pour un vrai niveau de 1v1.
