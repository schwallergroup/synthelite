echo "Setting up .env file..."

# Prompt the user for each variable
read -p "Enter your OPENROUTER_API_KEY: " openrouter_api_key
read -p "Enter your OPENAI_API_KEY: " openai_api_key
read -p "Enter your WANDB_PROJECT (e.g., synthelite): " wandb_project

# Write to .env file
cat > .env <<EOF
OPENROUTER_API_KEY=${openrouter_api_key}
OPENAI_API_KEY=${openai_api_key}
WANDB_PROJECT=${wandb_project}
EOF

echo ".env file created"

echo "Downloading necessary data..."
download_public_data data/