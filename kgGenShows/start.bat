@echo off
echo ======================================================
echo 知識圖譜可視化器 - 啟動腳本
echo ======================================================
echo.

echo 📋 檢查 Node.js 安裝...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 未檢測到 Node.js，請先安裝 Node.js
    echo 📥 下載地址：https://nodejs.org/
    pause
    exit /b 1
)

echo ✅ Node.js 已安裝
echo.

echo 📦 檢查依賴安裝...
if not exist "node_modules" (
    echo 🔧 首次運行，正在安裝依賴...
    npm install
    if %errorlevel% neq 0 (
        echo ❌ 依賴安裝失敗
        pause
        exit /b 1
    )
) else (
    echo ✅ 依賴已安裝
)

echo.
echo 🚀 啟動服務器...
echo ======================================================
echo 📍 服務器地址: http://localhost:3000
echo 📊 主要頁面: http://localhost:3000/
echo 🔍 簡易版本: http://localhost:3000/simple
echo ⚡ 完整版本: http://localhost:3000/full
echo ======================================================
echo 💡 提示: 按 Ctrl+C 停止服務器
echo.

npm start 