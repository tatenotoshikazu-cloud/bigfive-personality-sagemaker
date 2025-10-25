#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学生データの特徴量エンジニアリング
生データ → マッチング用スコアに変換
"""

import numpy as np
from typing import Dict, List

class StudentFeatureEngineer:
    """学生の生データをマッチング用特徴量に変換"""

    def __init__(self):
        # 大学ランクマスタ（偏差値ベース）
        self.university_tier = {
            '東京大学': 1.0, '京都大学': 1.0,
            '東京工業大学': 0.95, '一橋大学': 0.95,
            '大阪大学': 0.9, '東北大学': 0.9, '名古屋大学': 0.9,
            '早稲田大学': 0.85, '慶應義塾大学': 0.85,
            # ... 他の大学も定義
            'その他': 0.5  # デフォルト
        }

        # 専攻と業界の関連度マスタ
        self.major_industry_affinity = {
            '情報工学': {
                'IT_software': 1.0,
                'IT_data_science': 0.95,
                'consulting_strategy': 0.7,
                'finance_investment': 0.6,
            },
            '経済学': {
                'finance_investment': 1.0,
                'consulting_strategy': 0.9,
                'sales_b2b': 0.7,
            },
            # ... 他の専攻も定義
        }

    def transform(self, student_raw: Dict) -> Dict:
        """
        生データ → 特徴量スコアに変換

        Returns:
            {
                'personality_scores': {...},  # 性格スコア
                'competency_scores': {...},   # 能力スコア
                'motivation_scores': {...},   # 意欲スコア
                'industry_fit_scores': {...}, # 業界適合度
            }
        """

        # 1. 性格スコア（Big Five）
        personality_scores = self._calculate_personality_scores(
            student_raw['big5']
        )

        # 2. 能力スコア（学歴 + スキル + 経験）
        competency_scores = self._calculate_competency_scores(
            university=student_raw['university'],
            major=student_raw['major'],
            gpa=student_raw['gpa'],
            skills=student_raw['skills'],
            activities=student_raw['activities'],
            toeic=student_raw.get('toeic', 0)
        )

        # 3. 意欲スコア
        motivation_scores = self._calculate_motivation_scores(
            job_search_motivation=student_raw['job_search_motivation'],
            activities=student_raw['activities']
        )

        # 4. 業界適合度スコア（Big Five + 専攻 + スキルから総合判定）
        industry_fit_scores = self._calculate_industry_fit_scores(
            big5=student_raw['big5'],
            major=student_raw['major'],
            skills=student_raw['skills'],
            personality_scores=personality_scores,
            competency_scores=competency_scores
        )

        return {
            'personality_scores': personality_scores,
            'competency_scores': competency_scores,
            'motivation_scores': motivation_scores,
            'industry_fit_scores': industry_fit_scores,
        }

    def _calculate_personality_scores(self, big5: Dict) -> Dict:
        """性格スコア算出"""
        return {
            # そのまま使用
            'openness': big5['openness'],
            'conscientiousness': big5['conscientiousness'],
            'extraversion': big5['extraversion'],
            'agreeableness': big5['agreeableness'],
            'neuroticism': big5['neuroticism'],

            # 派生スコア
            'stress_tolerance': 1 - big5['neuroticism'],  # ストレス耐性
            'leadership_potential': (
                big5['extraversion'] * 0.5 +
                big5['conscientiousness'] * 0.3 +
                big5['openness'] * 0.2
            ),
            'teamwork_ability': (
                big5['agreeableness'] * 0.6 +
                big5['extraversion'] * 0.4
            ),
            'innovation_potential': (
                big5['openness'] * 0.7 +
                (1 - big5['neuroticism']) * 0.3
            )
        }

    def _calculate_competency_scores(
        self,
        university: str,
        major: str,
        gpa: float,
        skills: Dict,
        activities: Dict,
        toeic: int
    ) -> Dict:
        """能力スコア算出"""

        # 大学ティアスコア
        univ_score = self.university_tier.get(university, 0.5)

        # GPAスコア（4.0満点 → 0-1スケール）
        gpa_score = min(gpa / 4.0, 1.0)

        # 学業成績総合（大学ティア × GPA）
        academic_score = (univ_score * 0.6 + gpa_score * 0.4)

        # 技術スキルスコア（平均）
        tech_skill_score = np.mean(list(skills.values())) if skills else 0.5

        # 経験スコア
        experience_score = min((
            activities.get('internship_count', 0) * 0.15 +  # インターン（最大3回で0.45）
            (0.2 if activities.get('club_leader') else 0) +   # リーダー経験
            (0.1 if activities.get('volunteer') else 0) +     # ボランティア
            activities.get('hackathon', 0) * 0.1             # ハッカソン
        ), 1.0)

        # 語学力スコア（TOEIC 990満点 → 0-1スケール）
        language_score = min(toeic / 990, 1.0)

        return {
            'academic_level': academic_score,
            'technical_skill': tech_skill_score,
            'practical_experience': experience_score,
            'language_ability': language_score,

            # 総合能力
            'overall_competency': (
                academic_score * 0.3 +
                tech_skill_score * 0.3 +
                experience_score * 0.25 +
                language_score * 0.15
            )
        }

    def _calculate_motivation_scores(
        self,
        job_search_motivation: float,
        activities: Dict
    ) -> Dict:
        """意欲スコア算出"""

        # 就活意欲（そのまま）
        job_search_score = job_search_motivation

        # 成長意欲（活動履歴から推定）
        growth_mindset_score = min((
            activities.get('internship_count', 0) * 0.2 +
            activities.get('hackathon', 0) * 0.2 +
            (0.3 if activities.get('club_leader') else 0)
        ), 1.0)

        return {
            'job_search_motivation': job_search_score,
            'growth_mindset': growth_mindset_score,

            # 総合意欲
            'overall_motivation': (
                job_search_score * 0.6 +
                growth_mindset_score * 0.4
            )
        }

    def _calculate_industry_fit_scores(
        self,
        big5: Dict,
        major: str,
        skills: Dict,
        personality_scores: Dict,
        competency_scores: Dict
    ) -> Dict:
        """
        各業界への適合度スコアを算出
        ここが最も重要：Big Five + 学歴 + スキルを統合
        """

        industry_scores = {}

        # IT業界（ソフトウェア開発）
        industry_scores['IT_software'] = self._calc_IT_software_fit(
            big5, major, skills, personality_scores, competency_scores
        )

        # IT業界（データサイエンス）
        industry_scores['IT_data_science'] = self._calc_IT_datascience_fit(
            big5, major, skills, personality_scores, competency_scores
        )

        # 営業（B2B）
        industry_scores['sales_b2b'] = self._calc_sales_b2b_fit(
            big5, major, skills, personality_scores, competency_scores
        )

        # コンサルティング
        industry_scores['consulting_strategy'] = self._calc_consulting_fit(
            big5, major, skills, personality_scores, competency_scores
        )

        # ... 他の業界も同様に定義

        return industry_scores

    def _calc_IT_software_fit(
        self,
        big5: Dict,
        major: str,
        skills: Dict,
        personality_scores: Dict,
        competency_scores: Dict
    ) -> float:
        """IT・ソフトウェア開発への適合度"""

        # 1. 性格適合度（Big Five理想値との距離）
        ideal_big5 = [0.75, 0.75, 0.50, 0.55, 0.35]  # [O, C, E, A, N]
        student_big5 = [
            big5['openness'],
            big5['conscientiousness'],
            big5['extraversion'],
            big5['agreeableness'],
            big5['neuroticism']
        ]

        personality_distance = np.linalg.norm(
            np.array(student_big5) - np.array(ideal_big5)
        )
        personality_fit = 1 / (1 + personality_distance)

        # 2. 専攻適合度
        major_fit = self.major_industry_affinity.get(major, {}).get(
            'IT_software', 0.5
        )

        # 3. スキル適合度（Python, Java, SQLの平均）
        relevant_skills = ['Python', 'Java', 'SQL']
        skill_scores = [skills.get(s, 0) for s in relevant_skills]
        skill_fit = np.mean(skill_scores) if skill_scores else 0.5

        # 4. 能力レベル
        competency_fit = competency_scores['overall_competency']

        # 総合適合度（加重平均）
        total_fit = (
            personality_fit * 0.35 +      # 性格: 35%
            major_fit * 0.25 +            # 専攻: 25%
            skill_fit * 0.25 +            # スキル: 25%
            competency_fit * 0.15         # 能力: 15%
        )

        return total_fit

    def _calc_IT_datascience_fit(self, big5, major, skills, personality_scores, competency_scores):
        """IT・データサイエンスへの適合度"""

        ideal_big5 = [0.85, 0.70, 0.50, 0.55, 0.35]
        student_big5 = [big5['openness'], big5['conscientiousness'],
                       big5['extraversion'], big5['agreeableness'], big5['neuroticism']]

        personality_fit = 1 / (1 + np.linalg.norm(np.array(student_big5) - np.array(ideal_big5)))
        major_fit = self.major_industry_affinity.get(major, {}).get('IT_data_science', 0.5)

        # データサイエンスは数学・統計が重要
        relevant_skills = ['Python', 'SQL', 'R']
        skill_fit = np.mean([skills.get(s, 0) for s in relevant_skills])

        # 学業成績も重視（理論的素養）
        academic_fit = competency_scores['academic_level']

        return (
            personality_fit * 0.30 +
            major_fit * 0.20 +
            skill_fit * 0.30 +
            academic_fit * 0.20
        )

    def _calc_sales_b2b_fit(self, big5, major, skills, personality_scores, competency_scores):
        """営業（B2B）への適合度"""

        ideal_big5 = [0.55, 0.65, 0.80, 0.75, 0.35]
        student_big5 = [big5['openness'], big5['conscientiousness'],
                       big5['extraversion'], big5['agreeableness'], big5['neuroticism']]

        personality_fit = 1 / (1 + np.linalg.norm(np.array(student_big5) - np.array(ideal_big5)))

        # 営業はスキルより性格が重要
        # 外向性とストレス耐性を特に重視
        extraversion_bonus = big5['extraversion'] * 0.3
        stress_tolerance_bonus = personality_scores['stress_tolerance'] * 0.2

        return (
            personality_fit * 0.50 +
            extraversion_bonus +
            stress_tolerance_bonus
        )

    def _calc_consulting_fit(self, big5, major, skills, personality_scores, competency_scores):
        """コンサルティングへの適合度"""

        ideal_big5 = [0.80, 0.75, 0.75, 0.60, 0.35]
        student_big5 = [big5['openness'], big5['conscientiousness'],
                       big5['extraversion'], big5['agreeableness'], big5['neuroticism']]

        personality_fit = 1 / (1 + np.linalg.norm(np.array(student_big5) - np.array(ideal_big5)))

        # コンサルは学歴と論理的思考力が重要
        academic_fit = competency_scores['academic_level']

        # リーダーシップとイノベーション志向
        leadership_fit = personality_scores['leadership_potential']
        innovation_fit = personality_scores['innovation_potential']

        return (
            personality_fit * 0.30 +
            academic_fit * 0.25 +
            leadership_fit * 0.25 +
            innovation_fit * 0.20
        )


# 使用例
if __name__ == "__main__":
    # 学生の生データ
    student_raw = {
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

    # 特徴量エンジニアリング実行
    engineer = StudentFeatureEngineer()
    student_features = engineer.transform(student_raw)

    print("=" * 80)
    print("学生特徴量スコア")
    print("=" * 80)

    print("\n【性格スコア】")
    for key, val in student_features['personality_scores'].items():
        print(f"  {key:25s}: {val:.3f}")

    print("\n【能力スコア】")
    for key, val in student_features['competency_scores'].items():
        print(f"  {key:25s}: {val:.3f}")

    print("\n【意欲スコア】")
    for key, val in student_features['motivation_scores'].items():
        print(f"  {key:25s}: {val:.3f}")

    print("\n【業界適合度スコア】")
    for industry, score in sorted(
        student_features['industry_fit_scores'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"  {industry:30s}: {score:.3f} ({score*100:.1f}%)")
