#!/bin/bash
docker system prune -a
docker compose down
echo "LLM service stopped"
