# Node.js 版本安裝和使用說明

## 🚀 快速開始

### 1. 系統需求
- **Node.js**: 版本 14.0.0 或更高
- **npm**: Node.js 附帶的包管理器
- **瀏覽器**: Chrome 80+、Firefox 75+、Safari 13+、Edge 80+

### 2. 安裝步驟

#### 步驟 1: 確認 Node.js 安裝
在命令列中執行：
```bash
node --version
npm --version
```

如果顯示版本號，說明已安裝。否則請到 [Node.js 官網](https://nodejs.org/) 下載安裝。

#### 步驟 2: 進入專案資料夾
```bash
cd Miscellaneous/KgGen/kgGenShows
```

#### 步驟 3: 安裝相依套件
```bash
npm install
```

#### 步驟 4: 啟動伺服器
```bash
npm start
```

### 3. 訪問網站

伺服器啟動後，在瀏覽器中開啟：
- **主頁**: http://localhost:3000
- **簡易版**: http://localhost:3000/simple  
- **完整版**: http://localhost:3000/full
- **API**: http://localhost:3000/api/graph-data

## 📁 檔案結構

```
kgGenShows/
├── package.json                           # Node.js 專案配置
├── server.js                             # Express 伺服器主檔案
├── index.html                            # 主頁面
├── simple_graph_viewer.html              # 簡易版可視化器
├── knowledge_graph_visualizer.html       # 完整版可視化器
├── README.md                             # 原始說明文件
├── SETUP.md                              # 本安裝說明
└── ../final_results_20250619_133346.json # 數據源檔案
```

## 🛠️ 可用指令

### 基本指令
```bash
npm start           # 啟動生產環境伺服器
npm run dev         # 啟動開發環境伺服器（自動重啟）
npm run install-deps # 重新安裝相依套件
```

### 開發指令
```bash
# 啟動開發模式（需要先安裝 nodemon）
npm install -g nodemon
npm run dev
```

## 🌐 API 端點

### GET /api/graph-data
返回知識圖譜的完整數據
- **響應格式**: JSON
- **包含內容**: entities（實體列表）、relationships（關係列表）、metadata（元數據）

### GET /health
伺服器健康檢查
- **響應**: `{ "status": "healthy", "timestamp": "..." }`

## 🔧 故障排除

### 常見問題

#### 1. 端口被占用
```
Error: listen EADDRINUSE :::3000
```
**解決方法**:
- 方法 A: 更改端口
  ```bash
  PORT=3001 npm start
  ```
- 方法 B: 關閉占用端口的程序
  ```bash
  # Windows
  netstat -ano | findstr :3000
  taskkill /F /PID <PID>
  
  # macOS/Linux
  lsof -ti:3000 | xargs kill -9
  ```

#### 2. 相依套件安裝失敗
```bash
# 清除 npm 快取
npm cache clean --force

# 刪除 node_modules 重新安裝
rm -rf node_modules package-lock.json
npm install
```

#### 3. JSON 檔案找不到
確認 `final_results_20250619_133346.json` 位於正確路徑：
```
Miscellaneous/KgGen/final_results_20250619_133346.json
```

#### 4. 瀏覽器無法連接
- 確認伺服器已啟動
- 檢查防火牆設定
- 嘗試使用 `127.0.0.1:3000` 代替 `localhost:3000`

### 除錯模式

啟用詳細日誌：
```bash
DEBUG=* npm start
```

查看伺服器日誌：
```bash
# 伺服器會顯示詳細的請求信息
# 包括數據載入狀態、錯誤信息等
```

## 🎯 效能優化

### 針對大型數據集
如果遇到效能問題，可以調整以下參數：

1. **修改 server.js 中的緩存設定**
2. **調整 D3.js 力學模擬參數**
3. **使用簡易版本進行快速預覽**

### 記憶體使用優化
```bash
# 增加 Node.js 記憶體限制
node --max-old-space-size=4096 server.js
```

## 🔒 安全注意事項

### 開發環境
- 伺服器僅綁定 localhost，外部無法直接訪問
- 不建議在生產環境中使用此配置

### 生產部署
如需部署到生產環境，請考慮：
- 使用 PM2 或類似的程序管理器
- 配置 nginx 反向代理
- 啟用 HTTPS
- 設定適當的防火牆規則

## 📞 技術支援

### 日誌檢查
伺服器啟動時會顯示詳細信息：
```
🚀 知識圖譜可視化伺服器已啟動
📍 伺服器地址: http://localhost:3000
📊 主要頁面: http://localhost:3000/
🔍 簡易版本: http://localhost:3000/simple
⚡ 完整版本: http://localhost:3000/full
```

### 常用檢查指令
```bash
# 檢查 Node.js 版本
node --version

# 檢查端口使用情況
netstat -tulpn | grep :3000

# 檢查伺服器響應
curl http://localhost:3000/health
```

---

**建議**：初次使用請先嘗試簡易版本，確認數據載入正常後再使用完整版本。

**更新**：2025-07-05 - 支援 Node.js 環境，解決 CORS 問題 