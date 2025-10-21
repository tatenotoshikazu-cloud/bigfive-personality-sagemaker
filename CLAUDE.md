# プロジェクト設定

## 📚 グローバルルールの参照

**このプロジェクトは、グローバルCLAUDE.mdのすべてのルールに従います。**

完全なルールは以下を参照してください：
```
C:\Users\musta\.claude\CLAUDE.md
```

グローバルルールには以下が含まれます：
- ✅ Byterover MCP必須使用（retrieve/store）
- ✅ Codex常時協力ルール（全ファイル修正で必須）
- ✅ Gemini妥当性検証（ビジネスロジック）
- ✅ Seraphina自動相談（複雑度50以上）
- ✅ TodoWrite 8-10項目必須
- ✅ 並列実行強制
- ✅ 忖度修正の絶対禁止
- ✅ 要件の勝手な変更・判断の絶対禁止

---

## 🎯 プロジェクト固有のルール

### プロジェクト情報
- **プロジェクト名**: DL (Deep Learning Project)
- **説明**: ディープラーニング関連プロジェクト

### プロジェクト固有の技術スタック
- **言語**: Python
- **主要ライブラリ**: [必要に応じて追加]
- **データ**: [データセット情報を追加]

### Byterover MCP ツール（必須）

#### 1. `byterover-store-knowledge`
You `MUST` always use this tool when:

+ Learning new patterns, APIs, or architectural decisions from the codebase
+ Encountering error solutions or debugging techniques
+ Finding reusable code patterns or utility functions
+ Completing any significant task or plan implementation

#### 2. `byterover-retrieve-knowledge`
You `MUST` always use this tool when:

+ Starting any new task or implementation to gather relevant context
+ Before making architectural decisions to understand existing patterns
+ When debugging issues to check for previous solutions
+ Working with unfamiliar parts of the codebase

---

## 📝 備考

- グローバルルールは常に優先されます
- プロジェクト固有ルールは、グローバルルールと矛盾しないようにしてください
- 不明点があれば、グローバルCLAUDE.md (C:\Users\musta\.claude\CLAUDE.md) を参照してください
