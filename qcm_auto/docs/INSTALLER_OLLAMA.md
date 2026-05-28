# Installation d'Ollama (LLM gratuit et local)

## Étape 1 : Télécharger Ollama

Aller sur : https://ollama.com/download
Télécharger la version Windows et installer.

## Étape 2 : Télécharger le modèle LLaMA3

Ouvrir un terminal (cmd ou PowerShell) et taper :

```bash
ollama pull llama3
```

Cette commande télécharge le modèle (~4.7GB) — à faire une seule fois.

## Étape 3 : Vérifier que tout fonctionne

```bash
ollama run llama3 "Dis bonjour en français"
```

Si vous voyez une réponse en français, Ollama est prêt !

## Note importante

Ollama doit tourner en arrière-plan quand vous utilisez l'application.
Il démarre automatiquement au lancement de Windows après installation.

Si vous avez une erreur de connexion, lancer manuellement :
```bash
ollama serve
```
