#!/bin/bash

# --- Configuration ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Save project directory root
PROJECT_ROOT=$(pwd)

echo -e "${BLUE}=== Init Setup Project Environment ===${NC}"

# 1. Python 3.10 check
if ! command -v python3.10 &> /dev/null; then
    echo -e "${RED}Error: python3.10 not found!${NC}"
    echo "Please install it: brew install python@3.10"
    exit 1
fi

# 2. Manage venv
if [ -d ".venv" ]; then
    echo -e "${GREEN}✔ The virtual environment .venv already exists.${NC}"
else
    echo -e "${BLUE}Virtual environment creation with Python 3.10...${NC}"
    python3.10 -m venv .venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error during venv creation.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✔ .venv created.${NC}"
fi

# 3. Activation .venv
echo -e "${BLUE}Activation environment...${NC}"
source .venv/bin/activate

# 4. Upgrade Pip e Install Requirements
echo -e "${BLUE}Upgrade pip and requirements installation...${NC}"
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    mkdir -p ~/tmp_build
    TMPDIR=~/tmp_build pip install -r requirements.txt
    rm -rf ~/tmp_build
else
    echo -e "${RED}Attention: requirements.txt not found! Skip this step.${NC}"
fi

# 5. Back to Project Root
cd "$PROJECT_ROOT"

echo -e "${GREEN}=== Setup Completed Successfully! ===${NC}"
echo -e "Activate manually the virtual environment with: ${BLUE}source .venv/bin/activate${NC}"
