# -*- coding: utf-8 -*-
"""
大学偏差値マスタデータ

【データソース】
1. Claude AI 知識ベース（2025年1月時点）
2. Wikipedia「日本の大学一覧」（CC BY-SA 3.0）
   https://ja.wikipedia.org/wiki/日本の大学一覧
3. 各種公開情報を総合的に判断

【ライセンス】
- このデータはCC BY-SA 3.0ライセンスで提供されます
- 商用利用可、改変可、出典明記必須

【更新履歴】
- 2025-10-25: 初版作成（Claude AI + Wikipedia）

【注意事項】
- 偏差値は目安であり、学部・学科により異なります
- 年度により変動する可能性があります
- より正確なデータが必要な場合は公式情報を参照してください
"""

# ========================================
# 偏差値スコア定義（35-75スケール）
# ========================================

UNIVERSITY_DEVIATION_SCORES = {
    # ==========================================
    # Tier 1: 偏差値70以上（超トップ）
    # ==========================================
    '東京大学': 75,
    '京都大学': 74,
    '東京工業大学': 72,
    '一橋大学': 72,

    # ==========================================
    # Tier 2: 偏差値65-69（トップ国立・私立）
    # ==========================================
    '大阪大学': 68,
    '東北大学': 67,
    '名古屋大学': 67,
    '九州大学': 66,
    '北海道大学': 66,
    '神戸大学': 65,

    # トップ私立
    '慶應義塾大学': 68,
    '早稲田大学': 67,
    '上智大学': 65,
    '国際基督教大学': 65,

    # ==========================================
    # Tier 3: 偏差値60-64（MARCH、関関同立、準難関国立）
    # ==========================================

    # MARCH（明治・青山学院・立教・中央・法政）
    '明治大学': 62,
    '青山学院大学': 61,
    '立教大学': 62,
    '中央大学': 60,
    '法政大学': 60,

    # 関関同立（関西・関西学院・同志社・立命館）
    '同志社大学': 63,
    '関西学院大学': 61,
    '立命館大学': 60,
    '関西大学': 59,

    # 準難関国立
    '筑波大学': 63,
    '千葉大学': 62,
    '横浜国立大学': 62,
    '広島大学': 61,
    '岡山大学': 60,
    '金沢大学': 60,

    # その他有力私立
    '東京理科大学': 62,
    '学習院大学': 60,
    '津田塾大学': 60,
    '東京女子大学': 59,
    '日本女子大学': 59,

    # ==========================================
    # Tier 4: 偏差値55-59（中堅国立・成成明学獨國武）
    # ==========================================

    # 地方国立大学（駅弁大学）
    '静岡大学': 58,
    '新潟大学': 57,
    '信州大学': 57,
    '埼玉大学': 57,
    '三重大学': 56,
    '滋賀大学': 56,
    '長崎大学': 56,
    '熊本大学': 56,

    # 成成明学獨國武
    '成城大学': 58,
    '成蹊大学': 58,
    '明治学院大学': 57,
    '獨協大学': 56,
    '國學院大學': 56,
    '武蔵大学': 56,

    # 日東駒専（日本・東洋・駒澤・専修）
    '日本大学': 55,
    '東洋大学': 54,
    '駒澤大学': 53,
    '専修大学': 52,

    # 産近甲龍（京都産業・近畿・甲南・龍谷）
    '近畿大学': 55,
    '京都産業大学': 54,
    '甲南大学': 53,
    '龍谷大学': 52,

    # ==========================================
    # Tier 5: 偏差値50-54（中堅私立）
    # ==========================================
    '東海大学': 52,
    '亜細亜大学': 51,
    '帝京大学': 51,
    '国士舘大学': 50,
    '大東文化大学': 50,
    '東京経済大学': 50,
    '拓殖大学': 50,

    # ==========================================
    # Tier 6: 偏差値45-49（その他私立・地方国立）
    # ==========================================
    '桜美林大学': 48,
    '立正大学': 47,
    '大正大学': 46,
    '淑徳大学': 45,

    # ==========================================
    # デフォルト: 偏差値45（未定義大学）
    # ==========================================
    'その他': 45,
}


# ========================================
# ティア分類（グループ化）
# ========================================

UNIVERSITY_TIERS = {
    # Tier 1.0: 超トップ
    1.0: ['東京大学', '京都大学'],

    # Tier 0.95: トップ国立
    0.95: ['東京工業大学', '一橋大学'],

    # Tier 0.9: 旧帝大
    0.9: ['大阪大学', '東北大学', '名古屋大学', '九州大学', '北海道大学'],

    # Tier 0.85: 早慶
    0.85: ['慶應義塾大学', '早稲田大学'],

    # Tier 0.75: 準難関国立・上智ICU
    0.75: ['神戸大学', '筑波大学', '千葉大学', '横浜国立大学', '上智大学', '国際基督教大学'],

    # Tier 0.7: MARCH、関関同立
    0.7: ['明治大学', '青山学院大学', '立教大学', '中央大学', '法政大学',
          '同志社大学', '関西学院大学', '立命館大学', '関西大学'],

    # Tier 0.6: 地方国立、成成明学
    0.6: ['広島大学', '岡山大学', '金沢大学', '静岡大学', '新潟大学',
          '成城大学', '成蹊大学', '明治学院大学', '東京理科大学'],

    # Tier 0.5: 日東駒専、産近甲龍
    0.5: ['日本大学', '東洋大学', '駒澤大学', '専修大学',
          '近畿大学', '京都産業大学', '甲南大学', '龍谷大学'],

    # Tier 0.4: その他
    0.4: ['その他'],
}


# ========================================
# グループ別エイリアス（便利関数用）
# ========================================

UNIVERSITY_GROUPS = {
    'MARCH': ['明治大学', '青山学院大学', '立教大学', '中央大学', '法政大学'],
    '関関同立': ['関西大学', '関西学院大学', '同志社大学', '立命館大学'],
    '日東駒専': ['日本大学', '東洋大学', '駒澤大学', '専修大学'],
    '産近甲龍': ['京都産業大学', '近畿大学', '甲南大学', '龍谷大学'],
    '成成明学': ['成城大学', '成蹊大学', '明治学院大学'],
    '早慶': ['早稲田大学', '慶應義塾大学'],
    '旧帝大': ['東京大学', '京都大学', '大阪大学', '東北大学', '名古屋大学', '九州大学', '北海道大学'],
}


# ========================================
# ユーティリティ関数
# ========================================

def get_deviation_score(university_name: str) -> float:
    """
    大学名から偏差値スコアを取得

    Args:
        university_name: 大学名（例: '東京大学'）

    Returns:
        偏差値スコア（35-75）、未定義の場合は45

    Examples:
        >>> get_deviation_score('東京大学')
        75
        >>> get_deviation_score('明治大学')
        62
        >>> get_deviation_score('○○工業大学')
        45
    """
    return UNIVERSITY_DEVIATION_SCORES.get(university_name, UNIVERSITY_DEVIATION_SCORES['その他'])


def normalize_deviation_score(score: float) -> float:
    """
    偏差値スコア（35-75）を 0-1 スケールに正規化

    Args:
        score: 偏差値スコア（35-75）

    Returns:
        正規化スコア（0.0-1.0）

    Examples:
        >>> normalize_deviation_score(75)  # 東大
        1.0
        >>> normalize_deviation_score(62)  # MARCH
        0.675
        >>> normalize_deviation_score(45)  # その他
        0.25
    """
    MIN_SCORE = 35
    MAX_SCORE = 75
    return (score - MIN_SCORE) / (MAX_SCORE - MIN_SCORE)


def get_normalized_score(university_name: str) -> float:
    """
    大学名から正規化スコア（0-1）を直接取得

    Args:
        university_name: 大学名

    Returns:
        正規化スコア（0.0-1.0）

    Examples:
        >>> get_normalized_score('東京大学')
        1.0
        >>> get_normalized_score('明治大学')
        0.675
    """
    score = get_deviation_score(university_name)
    return normalize_deviation_score(score)


def get_tier_score(university_name: str) -> float:
    """
    大学名からティアスコア（0-1）を取得

    Args:
        university_name: 大学名

    Returns:
        ティアスコア（0.4-1.0）

    Examples:
        >>> get_tier_score('東京大学')
        1.0
        >>> get_tier_score('明治大学')
        0.7
    """
    for tier_value, universities in UNIVERSITY_TIERS.items():
        if university_name in universities:
            return tier_value
    return 0.4  # デフォルト


def is_in_group(university_name: str, group_name: str) -> bool:
    """
    大学が特定グループに属するか判定

    Args:
        university_name: 大学名
        group_name: グループ名（'MARCH', '関関同立', etc.）

    Returns:
        True if university is in the group, False otherwise

    Examples:
        >>> is_in_group('明治大学', 'MARCH')
        True
        >>> is_in_group('東京大学', 'MARCH')
        False
    """
    return university_name in UNIVERSITY_GROUPS.get(group_name, [])


def get_university_group(university_name: str) -> str:
    """
    大学が属するグループ名を取得

    Args:
        university_name: 大学名

    Returns:
        グループ名（該当なしの場合は'その他'）

    Examples:
        >>> get_university_group('明治大学')
        'MARCH'
        >>> get_university_group('東京大学')
        '旧帝大'
    """
    for group_name, universities in UNIVERSITY_GROUPS.items():
        if university_name in universities:
            return group_name
    return 'その他'


# ========================================
# データ検証・統計情報
# ========================================

def get_statistics():
    """
    マスタデータの統計情報を取得

    Returns:
        統計情報の辞書
    """
    scores = [v for k, v in UNIVERSITY_DEVIATION_SCORES.items() if k != 'その他']

    return {
        '大学数': len(scores),
        '最高偏差値': max(scores),
        '最低偏差値': min(scores),
        '平均偏差値': sum(scores) / len(scores),
        'Tier数': len(UNIVERSITY_TIERS),
        'グループ数': len(UNIVERSITY_GROUPS),
    }


def validate_data():
    """
    データの整合性チェック

    Returns:
        検証結果（問題なければTrue）
    """
    issues = []

    # 偏差値範囲チェック
    for univ, score in UNIVERSITY_DEVIATION_SCORES.items():
        if univ != 'その他' and not (35 <= score <= 75):
            issues.append(f'{univ}: 偏差値 {score} が範囲外（35-75）')

    # ティア定義の重複チェック
    all_tier_univs = []
    for universities in UNIVERSITY_TIERS.values():
        all_tier_univs.extend(universities)

    duplicates = [u for u in all_tier_univs if all_tier_univs.count(u) > 1]
    if duplicates:
        issues.append(f'ティア定義の重複: {set(duplicates)}')

    if issues:
        print('⚠️ データ検証エラー:')
        for issue in issues:
            print(f'  - {issue}')
        return False
    else:
        print('✅ データ検証OK')
        return True


# ========================================
# メイン実行（テスト・デモ）
# ========================================

if __name__ == '__main__':
    print('=' * 60)
    print('大学偏差値マスタデータ - 動作確認')
    print('=' * 60)

    # 統計情報表示
    print('\n📊 統計情報:')
    stats = get_statistics()
    for key, value in stats.items():
        print(f'  {key}: {value}')

    # データ検証
    print('\n🔍 データ検証:')
    validate_data()

    # 使用例
    print('\n📝 使用例:')
    test_universities = ['東京大学', '明治大学', '日本大学', '○○工業大学']

    for univ in test_universities:
        dev_score = get_deviation_score(univ)
        normalized = get_normalized_score(univ)
        tier = get_tier_score(univ)
        group = get_university_group(univ)

        print(f'\n  {univ}:')
        print(f'    偏差値: {dev_score}')
        print(f'    正規化スコア: {normalized:.3f}')
        print(f'    ティアスコア: {tier}')
        print(f'    グループ: {group}')

    print('\n' + '=' * 60)
    print('テスト完了')
    print('=' * 60)
