#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完全なマッチングシステム
学生の全データ（Big Five + 大学 + 活動 + 意欲）を使った企業マッチング
"""

from student_feature_engineering import StudentFeatureEngineer
from industry_big5_profiles import IndustryMatcher
import numpy as np
from typing import Dict, List


class ComprehensiveMatchingSystem:
    """
    学生のリッチなデータ × 企業の粗いデータ → マッチングスコア
    """

    def __init__(self):
        self.feature_engineer = StudentFeatureEngineer()
        self.industry_matcher = IndustryMatcher()

    def match_student_to_company(
        self,
        student_raw_data: Dict,
        company_data: Dict
    ) -> Dict:
        """
        メインのマッチング関数

        Args:
            student_raw_data: 学生の生データ
                {
                    'big5': {...},
                    'university': str,
                    'major': str,
                    'gpa': float,
                    'skills': {...},
                    'activities': {...},
                    'job_search_motivation': float,
                    'toeic': int
                }

            company_data: 企業データ（粗い）
                {
                    'company_id': str,
                    'company_name': str,
                    'industry_code': str  # これだけでOK！
                }

        Returns:
            {
                'total_score': float,  # 総合マッチングスコア（0-1）
                'breakdown': {...},    # スコア内訳
                'recommendation': str,  # レコメンド文
                'reasons': [...]       # マッチング理由リスト
            }
        """

        # Step 1: 学生データを特徴量スコアに変換
        student_features = self.feature_engineer.transform(student_raw_data)

        # Step 2: 学生の業界適合度スコアを取得
        student_industry_fit = student_features['industry_fit_scores']

        # Step 3: 企業の業界コードに対応する適合度を取得
        company_industry = company_data['industry_code']

        if company_industry not in student_industry_fit:
            # 業界コードが未定義の場合は、Big Fiveだけで計算
            basic_score = self.industry_matcher.calculate_matching_score(
                student_big5=list(student_raw_data['big5'].values()),
                industry_code=company_industry
            )
            industry_fit_score = basic_score['total_score']
        else:
            # 学生の業界適合度スコアを使用
            industry_fit_score = student_industry_fit[company_industry]

        # Step 4: 追加の調整要素を適用
        # (1) 就活意欲による調整
        motivation_boost = student_features['motivation_scores']['overall_motivation'] * 0.1

        # (2) 能力レベルによる調整
        competency_boost = student_features['competency_scores']['overall_competency'] * 0.1

        # Step 5: 総合スコア算出
        total_score = (
            industry_fit_score * 0.80 +  # 業界適合度: 80%
            motivation_boost +            # 意欲: 10%
            competency_boost              # 能力: 10%
        )

        # 0-1にクリッピング
        total_score = min(max(total_score, 0), 1)

        # Step 6: マッチング理由を生成
        reasons = self._generate_matching_reasons(
            student_features,
            company_industry,
            total_score
        )

        # Step 7: レコメンドメッセージ生成
        recommendation = self._generate_recommendation(
            total_score,
            reasons,
            company_data['company_name']
        )

        return {
            'company_id': company_data['company_id'],
            'company_name': company_data['company_name'],
            'industry_code': company_industry,
            'total_score': total_score,
            'breakdown': {
                'industry_fit': industry_fit_score,
                'motivation_boost': motivation_boost,
                'competency_boost': competency_boost
            },
            'student_features': student_features,
            'recommendation': recommendation,
            'reasons': reasons
        }

    def find_best_companies(
        self,
        student_raw_data: Dict,
        companies: List[Dict],
        top_k: int = 10
    ) -> List[Dict]:
        """
        学生に最適な企業をランキング

        Args:
            student_raw_data: 学生の生データ
            companies: 企業リスト
                [
                    {'company_id': '001', 'company_name': '株式会社A', 'industry_code': 'IT_software'},
                    {'company_id': '002', 'company_name': '株式会社B', 'industry_code': 'sales_b2b'},
                    ...
                ]
            top_k: 上位何社を返すか

        Returns:
            マッチングスコア降順の企業リスト
        """

        results = []

        for company in companies:
            match_result = self.match_student_to_company(
                student_raw_data,
                company
            )
            results.append(match_result)

        # スコア降順でソート
        results.sort(key=lambda x: x['total_score'], reverse=True)

        return results[:top_k]

    def _generate_matching_reasons(
        self,
        student_features: Dict,
        company_industry: str,
        total_score: float
    ) -> List[str]:
        """マッチング理由を生成"""

        reasons = []

        personality = student_features['personality_scores']
        competency = student_features['competency_scores']
        motivation = student_features['motivation_scores']

        # 性格面の強み
        if personality['leadership_potential'] >= 0.7:
            reasons.append(f"リーダーシップ資質が高い（{personality['leadership_potential']:.1%}）")

        if personality['teamwork_ability'] >= 0.7:
            reasons.append(f"チームワーク能力が高い（{personality['teamwork_ability']:.1%}）")

        if personality['innovation_potential'] >= 0.7:
            reasons.append(f"イノベーション志向が強い（{personality['innovation_potential']:.1%}）")

        if personality['stress_tolerance'] >= 0.7:
            reasons.append(f"ストレス耐性が高い（{personality['stress_tolerance']:.1%}）")

        # 能力面の強み
        if competency['academic_level'] >= 0.8:
            reasons.append(f"学業成績が優秀（{competency['academic_level']:.1%}）")

        if competency['technical_skill'] >= 0.7:
            reasons.append(f"技術スキルレベルが高い（{competency['technical_skill']:.1%}）")

        if competency['practical_experience'] >= 0.7:
            reasons.append(f"実務経験が豊富（{competency['practical_experience']:.1%}）")

        if competency['language_ability'] >= 0.7:
            reasons.append(f"語学力が高い（{competency['language_ability']:.1%}）")

        # 意欲面の強み
        if motivation['job_search_motivation'] >= 0.8:
            reasons.append(f"就職意欲が非常に高い（{motivation['job_search_motivation']:.1%}）")

        if motivation['growth_mindset'] >= 0.8:
            reasons.append(f"成長意欲が強い（{motivation['growth_mindset']:.1%}）")

        # 理由が少ない場合はデフォルトメッセージ
        if len(reasons) == 0:
            reasons.append("総合的なバランスが取れている")

        return reasons

    def _generate_recommendation(
        self,
        total_score: float,
        reasons: List[str],
        company_name: str
    ) -> str:
        """レコメンドメッセージ生成"""

        if total_score >= 0.85:
            level = "非常に高い"
            action = "積極的に応募をおすすめします"
        elif total_score >= 0.75:
            level = "高い"
            action = "応募を推奨します"
        elif total_score >= 0.65:
            level = "中程度"
            action = "検討をおすすめします"
        elif total_score >= 0.55:
            level = "やや低い"
            action = "他の選択肢も検討してください"
        else:
            level = "低い"
            action = "他の企業を優先してください"

        msg = f"{company_name}とのマッチング度: {total_score:.1%}（{level}）\n"
        msg += f"{action}\n\n"
        msg += "【主な理由】\n"
        for i, reason in enumerate(reasons[:5], 1):
            msg += f"  {i}. {reason}\n"

        return msg


# 使用例
if __name__ == "__main__":
    # 学生の生データ
    student = {
        'big5': {
            'openness': 0.72,
            'conscientiousness': 0.55,
            'extraversion': 0.88,
            'agreeableness': 0.41,
            'neuroticism': 0.34
        },
        'university': '東京大学',
        'major': '情報工学',
        'gpa': 3.5,
        'skills': {
            'Python': 0.8,
            'Java': 0.6,
            'SQL': 0.7,
            'React': 0.5
        },
        'activities': {
            'internship_count': 3,
            'club_leader': True,
            'volunteer': True,
            'hackathon': 2
        },
        'job_search_motivation': 0.85,
        'toeic': 850
    }

    # 企業データ（粗い情報のみ）
    companies = [
        {
            'company_id': '001',
            'company_name': '株式会社TechForward',
            'industry_code': 'IT_software'
        },
        {
            'company_id': '002',
            'company_name': '株式会社DataVision',
            'industry_code': 'IT_data_science'
        },
        {
            'company_id': '003',
            'company_name': '株式会社SalesPro',
            'industry_code': 'sales_b2b'
        },
        {
            'company_id': '004',
            'company_name': '株式会社StrategyConsult',
            'industry_code': 'consulting_strategy'
        },
    ]

    # マッチングシステム実行
    matcher = ComprehensiveMatchingSystem()

    print("=" * 80)
    print("学生・企業マッチングシステム - 実行例")
    print("=" * 80)

    # 単一企業とのマッチング
    print("\n【例1: 単一企業とのマッチング】")
    print("-" * 80)

    result = matcher.match_student_to_company(student, companies[0])

    print(f"\n企業名: {result['company_name']}")
    print(f"業界: {result['industry_code']}")
    print(f"\n総合マッチングスコア: {result['total_score']:.1%}")
    print(f"\nスコア内訳:")
    print(f"  業界適合度: {result['breakdown']['industry_fit']:.1%}")
    print(f"  意欲ボーナス: +{result['breakdown']['motivation_boost']:.1%}")
    print(f"  能力ボーナス: +{result['breakdown']['competency_boost']:.1%}")
    print(f"\n{result['recommendation']}")

    # 複数企業のランキング
    print("\n" + "=" * 80)
    print("【例2: 最適企業ランキング】")
    print("=" * 80)

    top_companies = matcher.find_best_companies(student, companies, top_k=4)

    for i, result in enumerate(top_companies, 1):
        print(f"\n第{i}位: {result['company_name']}")
        print(f"  業界: {result['industry_code']}")
        print(f"  マッチング度: {result['total_score']:.1%}")
        print(f"  主な理由:")
        for reason in result['reasons'][:3]:
            print(f"    - {reason}")
