#!/bin/bash

# Professional Crypto Trading Bot Installation Script
# Compatible with cp.springhost.ru and Python 3.13.1

echo "🤖 Installing Professional Crypto Trading Bot..."
echo "================================================"

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Python 3.13.1 if not available
echo "🐍 Checking Python version..."
python3 --version

# Install pip if not available
echo "📦 Installing pip..."
sudo apt-get install python3-pip -y

# Install system dependencies for matplotlib and other packages
echo "🔧 Installing system dependencies..."
sudo apt-get install -y \
    python3-dev \
    python3-setuptools \
    build-essential \
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    libffi-dev \
    libssl-dev \
    pkg-config \
    gcc \
    g++ \
    make \
    cmake \
    git \
    curl \
    wget \
    unzip

# Upgrade pip to latest version
echo "⬆️ Upgrading pip..."
python3 -m pip install --upgrade pip

# Install Python packages
echo "📚 Installing Python packages..."
python3 -m pip install --upgrade \
    ccxt==4.2.25 \
    pandas==2.1.4 \
    numpy==1.26.2 \
    ta==0.10.2 \
    matplotlib==3.8.2 \
    seaborn==0.13.0 \
    python-telegram-bot==20.7 \
    reportlab==4.0.8 \
    requests==2.31.0 \
    beautifulsoup4==4.12.2 \
    aiohttp \
    asyncio-throttle \
    python-dateutil \
    pytz \
    lxml \
    Pillow

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p charts
mkdir -p data
mkdir -p backups

# Set permissions
echo "🔐 Setting permissions..."
chmod +x index.py
chmod 755 logs charts data backups

# Create systemd service file for auto-start
echo "⚙️ Creating systemd service..."
sudo tee /etc/systemd/system/trading-bot.service > /dev/null <<EOF
[Unit]
Description=Professional Crypto Trading Bot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=/usr/bin:/usr/local/bin
Environment=PYTHONPATH=$(pwd)
ExecStart=/usr/bin/python3 $(pwd)/index.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable trading-bot.service

# Create backup script
echo "💾 Creating backup script..."
tee backup.sh > /dev/null <<EOF
#!/bin/bash
# Backup script for trading bot data

DATE=\$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/backup_\$DATE"

mkdir -p \$BACKUP_DIR

# Backup database and logs
cp trading_bot.db \$BACKUP_DIR/ 2>/dev/null || echo "No database found"
cp trading_bot.log \$BACKUP_DIR/ 2>/dev/null || echo "No log file found"
cp -r charts \$BACKUP_DIR/ 2>/dev/null || echo "No charts found"

# Compress backup
tar -czf \$BACKUP_DIR.tar.gz \$BACKUP_DIR
rm -rf \$BACKUP_DIR

echo "Backup created: \$BACKUP_DIR.tar.gz"

# Keep only last 10 backups
ls -t backups/*.tar.gz | tail -n +11 | xargs -r rm

echo "Backup completed successfully!"
EOF

chmod +x backup.sh

# Create update script
echo "🔄 Creating update script..."
tee update.sh > /dev/null <<EOF
#!/bin/bash
# Update script for trading bot

echo "🔄 Updating Trading Bot..."

# Stop the service
sudo systemctl stop trading-bot.service

# Backup current version
./backup.sh

# Update Python packages
python3 -m pip install --upgrade \
    ccxt \
    pandas \
    numpy \
    ta \
    matplotlib \
    seaborn \
    python-telegram-bot \
    reportlab \
    requests \
    beautifulsoup4

# Restart the service
sudo systemctl start trading-bot.service
sudo systemctl status trading-bot.service

echo "✅ Update completed!"
EOF

chmod +x update.sh

# Create monitoring script
echo "📊 Creating monitoring script..."
tee monitor.sh > /dev/null <<EOF
#!/bin/bash
# Monitoring script for trading bot

echo "🤖 Trading Bot Status Monitor"
echo "============================="

# Check service status
echo "📊 Service Status:"
sudo systemctl status trading-bot.service --no-pager -l

echo ""
echo "📈 Resource Usage:"
ps aux | grep "python3.*index.py" | grep -v grep

echo ""
echo "💾 Disk Usage:"
df -h .

echo ""
echo "📝 Recent Logs (last 20 lines):"
tail -n 20 trading_bot.log 2>/dev/null || echo "No log file found"

echo ""
echo "🗄️ Database Size:"
ls -lh trading_bot.db 2>/dev/null || echo "No database found"
EOF

chmod +x monitor.sh

# Create start/stop scripts
echo "▶️ Creating control scripts..."
tee start.sh > /dev/null <<EOF
#!/bin/bash
echo "🚀 Starting Trading Bot..."
sudo systemctl start trading-bot.service
sudo systemctl status trading-bot.service --no-pager
EOF

tee stop.sh > /dev/null <<EOF
#!/bin/bash
echo "🛑 Stopping Trading Bot..."
sudo systemctl stop trading-bot.service
echo "Bot stopped successfully!"
EOF

tee restart.sh > /dev/null <<EOF
#!/bin/bash
echo "🔄 Restarting Trading Bot..."
sudo systemctl restart trading-bot.service
sudo systemctl status trading-bot.service --no-pager
EOF

chmod +x start.sh stop.sh restart.sh

# Test Python imports
echo "🧪 Testing Python imports..."
python3 -c "
import ccxt
import pandas as pd
import numpy as np
import ta
import matplotlib
import telegram
import reportlab
import requests
import sqlite3
print('✅ All imports successful!')
"

# Create configuration template
echo "⚙️ Creating configuration template..."
tee config_template.py > /dev/null <<EOF
# Configuration Template for Trading Bot
# Copy this file to config.py and modify as needed

TELEGRAM_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"

# Trading pairs to monitor
SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT",
    "DOGE/USDT", "SHIB/USDT", "PEPE/USDT", "BONK/USDT",
    "SOL/USDT", "MATIC/USDT", "AVAX/USDT", "DOT/USDT"
]

# Risk management settings
RISK_REWARD_RATIO = 2.5
STOP_LOSS_ATR_MULTIPLIER = 2.0
TAKE_PROFIT_ATR_MULTIPLIER = 4.0

# Analysis intervals (minutes)
ANALYSIS_INTERVAL = 15
ALERT_CHECK_INTERVAL = 5
EOF

echo ""
echo "🎉 Installation completed successfully!"
echo "======================================"
echo ""
echo "📋 Next steps:"
echo "1. Edit the Telegram token and chat ID in index.py"
echo "2. Test the bot: python3 index.py"
echo "3. Start the service: ./start.sh"
echo "4. Monitor the bot: ./monitor.sh"
echo ""
echo "🔧 Available commands:"
echo "• ./start.sh     - Start the bot service"
echo "• ./stop.sh      - Stop the bot service"  
echo "• ./restart.sh   - Restart the bot service"
echo "• ./monitor.sh   - Monitor bot status"
echo "• ./backup.sh    - Create backup"
echo "• ./update.sh    - Update bot and packages"
echo ""
echo "📊 The bot will automatically:"
echo "• Analyze markets every 15 minutes"
echo "• Send signals to your Telegram"
echo "• Generate daily reports"
echo "• Create professional charts"
echo "• Track performance"
echo ""
echo "⚠️  Remember to:"
echo "• Configure your Telegram bot token"
echo "• Set up your chat ID"
echo "• Test thoroughly before live trading"
echo "• Never risk more than you can afford to lose"
echo ""
echo "🚀 Happy trading!"
EOF
