# Python 除錯問題預防指南

## 🎯 設計一致性預防

### 問題類型：架構不一致
```python
# ❌ 錯誤：測試期望組合模式，但實現使用靜態方法
# 測試中：judge.prompt_engineer.create_prompt()
# 實現中：PromptEngineer.create_prompt()
```

### 預防措施
1. **TDD 原則** - 先寫測試，確定介面設計
2. **架構文檔** - 明確定義類關係（組合/繼承/靜態）
3. **一致性檢查** - 確保測試和實現使用相同模式

## 🔄 Python 緩存問題預防

### 問題類型：模組緩存導致修改不生效
```bash
# 症狀：代碼已修改但運行時使用舊版本
# 原因：Python 的 .pyc 緓存和 sys.modules 緩存
```

### 預防措施
1. **開發環境設置**
```bash
# 在 .gitignore 中添加
__pycache__/
*.pyc
*.pyo
.pytest_cache/

# 開發時定期清理
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +
```

2. **強制重載設置**
```python
# conftest.py 中添加
import importlib
import sys

def pytest_configure(config):
    """強制重載所有項目模組"""
    for module_name in list(sys.modules.keys()):
        if 'your_project_name' in module_name:
            sys.modules.pop(module_name, None)
```

3. **IDE 設置**
- PyCharm: Settings → Build → Compiler → Clear cache and restart
- VSCode: 重啟 Python 解釋器（Ctrl+Shift+P → Python: Restart Language Server）

## ⚡ 初始化順序問題預防

### 問題類型：關鍵屬性在異常前未初始化
```python
# ❌ 錯誤順序
def __init__(self):
    self.basic_attr = value
    if not api_key:  # 這裡可能拋出異常
        raise ValueError("Missing API key")
    self.important_attr = SomeClass()  # 永遠不會執行
```

### 預防措施
```python
# ✅ 正確順序
def __init__(self):
    try:
        # 1. 初始化所有基本屬性
        self.basic_attr = value
        self.important_attr = SomeClass()  # 確保創建
        
        # 2. 然後進行可能失敗的驗證
        if not api_key and not mock_mode:
            raise ValueError("Missing API key")
    except Exception as e:
        # 3. 異常處理中記錄已初始化的屬性
        print(f"Init failed, current attrs: {list(self.__dict__.keys())}")
        raise
```

## 🐛 調試策略改進

### 問題類型：錯誤信息不足，難以定位問題

### 預防措施
1. **結構化調試信息**
```python
class DebugMixin:
    def __init__(self):
        self._debug_enabled = os.getenv('DEBUG', 'false').lower() == 'true'
    
    def debug_print(self, stage: str, message: str):
        if self._debug_enabled:
            print(f"🔧 DEBUG [{self.__class__.__name__}:{stage}]: {message}")

class YourClass(DebugMixin):
    def __init__(self):
        super().__init__()
        self.debug_print("init", "Starting initialization")
        # ... 其他初始化代碼
```

2. **關鍵檢查點**
```python
def critical_method(self):
    # 前置條件檢查
    assert hasattr(self, 'required_attr'), f"Missing required_attr in {self.__class__.__name__}"
    
    # 執行業務邏輯
    result = self.do_something()
    
    # 後置條件檢查
    assert result is not None, "Method returned None unexpectedly"
    return result
```

## 🧪 測試策略改進

### 問題類型：測試無法有效捕獲實際使用中的問題

### 預防措施
1. **多層次測試**
```python
# 單元測試：測試個別方法
def test_prompt_engineer_creation():
    engineer = PromptEngineer()
    assert engineer is not None

# 整合測試：測試類之間的交互
def test_judge_with_prompt_engineer():
    judge = PerplexityGraphJudge()
    assert hasattr(judge, 'prompt_engineer')
    prompt = judge.prompt_engineer.create_prompt("test")
    assert "test" in prompt

# 端到端測試：測試完整流程
async def test_full_workflow():
    judge = PerplexityGraphJudge()
    result = await judge.judge_graph_triple("Is this true: test ?")
    assert result in ["Yes", "No"]
```

2. **屬性存在性測試**
```python
def test_required_attributes():
    """確保所有必需的屬性都存在"""
    judge = PerplexityGraphJudge()
    
    required_attrs = ['model_name', 'prompt_engineer', 'is_mock']
    for attr in required_attrs:
        assert hasattr(judge, attr), f"Missing required attribute: {attr}"
        assert getattr(judge, attr) is not None, f"Attribute {attr} is None"
```

## 📋 除錯 Checklist

遇到類似問題時，按順序檢查：

### ✅ 立即檢查
- [ ] 清除所有 Python 緩存 (`rm -rf __pycache__ *.pyc`)
- [ ] 重啟 IDE/編輯器
- [ ] 確認文件已保存到磁碟

### ✅ 架構檢查
- [ ] 測試中的調用方式與實現一致
- [ ] 所有必需屬性在 `__init__` 中正確初始化
- [ ] 初始化順序：基本屬性 → 可能失敗的驗證

### ✅ 調試信息
- [ ] 添加詳細的調試打印
- [ ] 檢查 `hasattr()` 和 `dir()` 輸出
- [ ] 記錄初始化過程中的每個步驟

### ✅ 測試覆蓋
- [ ] 單元測試覆蓋個別方法
- [ ] 整合測試覆蓋類交互
- [ ] 屬性存在性測試

## 🚀 快速修復腳本

創建一個快速診斷腳本：

```python
#!/usr/bin/env python3
"""quick_debug.py - 快速診斷腳本"""

import sys
import os

def clear_cache():
    """清除 Python 緩存"""
    os.system("find . -name '*.pyc' -delete 2>/dev/null || true")
    os.system("find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true")
    print("✅ 緩存已清除")

def check_imports():
    """檢查關鍵模組導入"""
    try:
        from your_module import YourClass
        print("✅ 模組導入成功")
        return YourClass
    except Exception as e:
        print(f"❌ 導入失敗: {e}")
        return None

def check_attributes(cls):
    """檢查類屬性"""
    try:
        instance = cls()
        attrs = list(instance.__dict__.keys())
        print(f"✅ 實例屬性: {attrs}")
        return instance
    except Exception as e:
        print(f"❌ 實例化失敗: {e}")
        return None

if __name__ == "__main__":
    clear_cache()
    cls = check_imports()
    if cls:
        check_attributes(cls)
```

## 📝 最佳實踐總結

1. **設計階段**：確保測試和實現的一致性
2. **開發階段**：定期清除緩存，使用結構化調試
3. **測試階段**：多層次測試，包含屬性存在性測試
4. **除錯階段**：遵循系統性檢查清單

記住：**預防勝於治療**，良好的開發習慣可以避免大部分除錯問題。
