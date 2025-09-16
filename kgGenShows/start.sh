#!/bin/bash

echo "======================================================"
echo "知識圖譜可視化器 - 啟動腳本"
echo "======================================================"
echo

echo "📋 檢查 Node.js 安裝..."
if ! command -v node &> /dev/null; then
    echo "❌ 未檢測到 Node.js，請先安裝 Node.js"
    echo "📥 下載地址：https://nodejs.org/"
    exit 1
fi

echo "✅ Node.js 已安裝 ($(node --version))"
echo

echo "📦 檢查依賴安裝..."
if [ ! -d "node_modules" ]; then
    echo "🔧 首次運行，正在安裝依賴..."
    npm install
    if [ $? -ne 0 ]; then
        echo "❌ 依賴安裝失敗"
        exit 1
    fi
else
    echo "✅ 依賴已安裝"
fi

echo
echo "🚀 啟動服務器..."
echo "======================================================"
echo "📍 服務器地址: http://localhost:3000"
echo "📊 主要頁面: http://localhost:3000/"
echo "🔍 簡易版本: http://localhost:3000/simple"
echo "⚡ 完整版本: http://localhost:3000/full"
echo "======================================================"
echo "💡 提示: 按 Ctrl+C 停止服務器"
echo

npm start 