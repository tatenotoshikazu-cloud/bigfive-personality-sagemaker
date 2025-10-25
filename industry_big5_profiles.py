#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
業界別 Big Five 理想プロファイル辞書

学術研究に基づく業界ごとの性格特性傾向
出典: Judge et al. (2002), Barrick & Mount (1991), Holland's RIASEC
"""

import numpy as np
from typing import Dict, List

# 業界別 Big Five 理想プロファイル（0-1スケール）
INDUSTRY_BIG5_PROFILES = {
    # IT・テクノロジー
    "IT_software_development": {
        "name": "IT・ソフトウェア開発",
        "big5_ideal": [0.75, 0.75, 0.50, 0.55, 0.35],  # [O, C, E, A, N]
        "description": "新技術への適応力と正確性が求められる",
        "key_traits": ["開放性", "誠実性"],
    },
    "IT_infrastructure": {
        "name": "IT・インフラ/ネットワーク",
        "big5_ideal": [0.60, 0.85, 0.45, 0.60, 0.30],
        "description": "高い誠実性と安定性が重要",
        "key_traits": ["誠実性", "協調性"],
    },
    "IT_data_science": {
        "name": "IT・データサイエンス/AI",
        "big5_ideal": [0.85, 0.70, 0.50, 0.55, 0.35],
        "description": "分析力と創造性の両立",
        "key_traits": ["開放性", "誠実性"],
    },

    # 営業・マーケティング
    "sales_b2b": {
        "name": "営業・法人向け（B2B）",
        "big5_ideal": [0.55, 0.65, 0.80, 0.75, 0.35],
        "description": "高い外向性とストレス耐性",
        "key_traits": ["外向性", "協調性"],
    },
    "sales_b2c": {
        "name": "営業・個人向け（B2C）",
        "big5_ideal": [0.60, 0.60, 0.85, 0.80, 0.30],
        "description": "共感力とコミュニケーション能力",
        "key_traits": ["外向性", "協調性"],
    },
    "marketing": {
        "name": "マーケティング",
        "big5_ideal": [0.75, 0.60, 0.70, 0.65, 0.40],
        "description": "創造性とデータ分析の両立",
        "key_traits": ["開放性", "外向性"],
    },

    # クリエイティブ
    "creative_design": {
        "name": "クリエイティブ・デザイン",
        "big5_ideal": [0.90, 0.55, 0.60, 0.60, 0.50],
        "description": "高い創造性と美的感覚",
        "key_traits": ["開放性"],
    },
    "creative_advertising": {
        "name": "クリエイティブ・広告",
        "big5_ideal": [0.85, 0.60, 0.75, 0.65, 0.45],
        "description": "創造性とプレゼンテーション能力",
        "key_traits": ["開放性", "外向性"],
    },

    # 製造・生産
    "manufacturing_production": {
        "name": "製造・生産管理",
        "big5_ideal": [0.45, 0.90, 0.45, 0.60, 0.30],
        "description": "極めて高い誠実性と安全意識",
        "key_traits": ["誠実性"],
    },
    "manufacturing_quality": {
        "name": "製造・品質管理",
        "big5_ideal": [0.50, 0.95, 0.40, 0.65, 0.25],
        "description": "完璧主義と細部への注意",
        "key_traits": ["誠実性"],
    },

    # 金融・会計
    "finance_investment": {
        "name": "金融・投資",
        "big5_ideal": [0.65, 0.80, 0.55, 0.50, 0.35],
        "description": "分析力とリスク管理",
        "key_traits": ["開放性", "誠実性"],
    },
    "finance_accounting": {
        "name": "金融・会計",
        "big5_ideal": [0.50, 0.90, 0.45, 0.60, 0.30],
        "description": "正確性と法令遵守",
        "key_traits": ["誠実性"],
    },

    # コンサルティング
    "consulting_strategy": {
        "name": "コンサルティング・戦略",
        "big5_ideal": [0.80, 0.75, 0.75, 0.60, 0.35],
        "description": "分析力とプレゼンテーション",
        "key_traits": ["開放性", "外向性"],
    },
    "consulting_hr": {
        "name": "コンサルティング・人事",
        "big5_ideal": [0.70, 0.70, 0.75, 0.80, 0.40],
        "description": "共感力とコミュニケーション",
        "key_traits": ["外向性", "協調性"],
    },

    # 教育・研究
    "education_teaching": {
        "name": "教育・教員",
        "big5_ideal": [0.70, 0.70, 0.70, 0.85, 0.40],
        "description": "高い協調性と忍耐力",
        "key_traits": ["協調性", "外向性"],
    },
    "research_academic": {
        "name": "研究・学術",
        "big5_ideal": [0.90, 0.80, 0.40, 0.55, 0.45],
        "description": "探究心と集中力",
        "key_traits": ["開放性", "誠実性"],
    },

    # 医療・福祉
    "healthcare_medical": {
        "name": "医療・医師/看護師",
        "big5_ideal": [0.60, 0.85, 0.65, 0.85, 0.30],
        "description": "高い協調性と責任感",
        "key_traits": ["協調性", "誠実性"],
    },
    "healthcare_welfare": {
        "name": "医療・福祉/介護",
        "big5_ideal": [0.55, 0.75, 0.70, 0.90, 0.35],
        "description": "共感力と献身性",
        "key_traits": ["協調性"],
    },

    # サービス業
    "service_hospitality": {
        "name": "サービス・ホスピタリティ",
        "big5_ideal": [0.60, 0.70, 0.80, 0.85, 0.35],
        "description": "おもてなしの心とストレス耐性",
        "key_traits": ["外向性", "協調性"],
    },
    "service_customer_support": {
        "name": "サービス・カスタマーサポート",
        "big5_ideal": [0.55, 0.75, 0.70, 0.85, 0.30],
        "description": "忍耐力と問題解決能力",
        "key_traits": ["協調性", "誠実性"],
    },
}


class IndustryMatcher:
    """業界ベースのマッチングエンジン"""

    def __init__(self, profiles: Dict = None):
        self.profiles = profiles or INDUSTRY_BIG5_PROFILES

    def calculate_matching_score(
        self,
        student_big5: List[float],
        industry_code: str,
        weights: Dict[str, float] = None
    ) -> Dict:
        """
        学生と業界のマッチングスコア算出

        Args:
            student_big5: 学生のBig Five [O, C, E, A, N]
            industry_code: 業界コード（例: "IT_software_development"）
            weights: 各特性の重み（デフォルト: 均等）

        Returns:
            {
                'total_score': float,  # 総合スコア（0-1）
                'industry_name': str,
                'distance': float,  # L2距離
                'trait_scores': dict,  # 特性別スコア
                'recommendation': str  # レコメンドメッセージ
            }
        """
        if industry_code not in self.profiles:
            raise ValueError(f"Unknown industry code: {industry_code}")

        profile = self.profiles[industry_code]
        ideal_big5 = profile['big5_ideal']

        # デフォルト重み: 均等（各20%）
        if weights is None:
            weights = {
                'openness': 0.20,
                'conscientiousness': 0.20,
                'extraversion': 0.20,
                'agreeableness': 0.20,
                'neuroticism': 0.20,
            }

        # L2距離（ユークリッド距離）
        distance = np.linalg.norm(
            np.array(student_big5) - np.array(ideal_big5)
        )

        # 距離を類似度に変換（0-1スケール）
        # 距離0（完全一致）→ スコア1.0
        # 距離が大きいほどスコア低下
        similarity = 1 / (1 + distance)

        # 特性別スコア計算
        trait_names = ['openness', 'conscientiousness', 'extraversion',
                      'agreeableness', 'neuroticism']
        trait_scores = {}

        for i, trait in enumerate(trait_names):
            diff = abs(student_big5[i] - ideal_big5[i])
            trait_scores[trait] = {
                'student': student_big5[i],
                'ideal': ideal_big5[i],
                'diff': diff,
                'match_score': 1 - diff  # 差が小さいほど高スコア
            }

        # レコメンドメッセージ生成
        key_traits = profile.get('key_traits', [])
        recommendation = self._generate_recommendation(
            similarity, trait_scores, key_traits
        )

        return {
            'total_score': similarity,
            'industry_code': industry_code,
            'industry_name': profile['name'],
            'distance': distance,
            'trait_scores': trait_scores,
            'description': profile['description'],
            'key_traits': key_traits,
            'recommendation': recommendation
        }

    def find_best_industries(
        self,
        student_big5: List[float],
        top_k: int = 5
    ) -> List[Dict]:
        """
        学生に最適な業界をランキング

        Args:
            student_big5: 学生のBig Five
            top_k: 上位何件を返すか

        Returns:
            マッチングスコア降順のリスト
        """
        results = []

        for industry_code in self.profiles.keys():
            score_result = self.calculate_matching_score(
                student_big5, industry_code
            )
            results.append(score_result)

        # スコア降順でソート
        results.sort(key=lambda x: x['total_score'], reverse=True)

        return results[:top_k]

    def _generate_recommendation(
        self,
        score: float,
        trait_scores: Dict,
        key_traits: List[str]
    ) -> str:
        """レコメンドメッセージ生成"""

        trait_map = {
            '開放性': 'openness',
            '誠実性': 'conscientiousness',
            '外向性': 'extraversion',
            '協調性': 'agreeableness',
            '神経症傾向': 'neuroticism'
        }

        if score >= 0.85:
            msg = "非常に高い適合性があります。"
        elif score >= 0.75:
            msg = "高い適合性があります。"
        elif score >= 0.65:
            msg = "適合性があります。"
        elif score >= 0.55:
            msg = "一部適合性がありますが、要検討。"
        else:
            msg = "適合性が低い可能性があります。"

        # 重要な特性の一致度を追加
        strengths = []
        weaknesses = []

        for jp_trait in key_traits:
            en_trait = trait_map.get(jp_trait)
            if en_trait:
                match_score = trait_scores[en_trait]['match_score']
                if match_score >= 0.8:
                    strengths.append(jp_trait)
                elif match_score < 0.6:
                    weaknesses.append(jp_trait)

        if strengths:
            msg += f" 強み: {', '.join(strengths)}が適合。"
        if weaknesses:
            msg += f" 要改善: {', '.join(weaknesses)}の適合度向上推奨。"

        return msg


# 使用例
if __name__ == "__main__":
    # Big Five予測結果（例）
    student = [0.72, 0.55, 0.88, 0.41, 0.34]  # [O, C, E, A, N]

    matcher = IndustryMatcher()

    # 特定業界とのマッチング
    print("=" * 80)
    print("特定業界とのマッチング")
    print("=" * 80)

    result = matcher.calculate_matching_score(
        student, "sales_b2b"
    )

    print(f"\n業界: {result['industry_name']}")
    print(f"説明: {result['description']}")
    print(f"総合スコア: {result['total_score']:.3f}")
    print(f"レコメンド: {result['recommendation']}")
    print("\n特性別詳細:")
    for trait, scores in result['trait_scores'].items():
        print(f"  {trait:20s}: 学生={scores['student']:.2f}, "
              f"理想={scores['ideal']:.2f}, "
              f"一致度={scores['match_score']:.2f}")

    # 最適業界ランキング
    print("\n" + "=" * 80)
    print("最適業界ランキング TOP 5")
    print("=" * 80)

    top_industries = matcher.find_best_industries(student, top_k=5)

    for i, result in enumerate(top_industries, 1):
        print(f"\n第{i}位: {result['industry_name']}")
        print(f"  スコア: {result['total_score']:.3f}")
        print(f"  説明: {result['description']}")
        print(f"  重要特性: {', '.join(result['key_traits'])}")
