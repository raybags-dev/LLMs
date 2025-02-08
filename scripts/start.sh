#!/bin/bash
docker compose up -d
echo "LLM service started"
docker logs -f llm-service
