import pandas as pd
import random
from datetime import datetime, timedelta
import json
import os

class AnchorGenerator:
    def __init__(self, seed=42):
        random.seed(seed)
        self.arms = ["Placebo", "Xanomeline High Dose", "Xanomeline Low Dose"]

    def generate_date(self, start_date, range_days):
        days = random.randint(0, range_days)
        return start_date + timedelta(days=days)

    def generate_anchors(self, config=None):
        if config is None:
            config = {
                "n_subjects": 10,
                "age_range": (18, 75),
                "gender_ratio": {"M": 0.5, "F": 0.5}, # Approximate probability
                "arms": ["Placebo", "Xanomeline High Dose", "Xanomeline Low Dose"],
                "start_date": "2023-01-01"
            }
        
        anchors = []
        base_date = datetime.strptime(config.get("start_date", "2023-01-01"), "%Y-%m-%d")
        arms = config.get("arms", ["Placebo"])
        n_subjects = config.get("n_subjects", 10)
        age_min, age_max = config.get("age_range", (18, 75))
        
        # Gender logic
        gender_probs = config.get("gender_ratio", {"M": 0.5, "F": 0.5})
        genders = list(gender_probs.keys())
        weights = list(gender_probs.values())

        for i in range(1, n_subjects + 1):
            usubjid = f"01-701-{1000+i}"
            arm = random.choice(arms)
            
            # Demographics
            age = random.randint(age_min, age_max)
            sex = random.choices(genders, weights=weights, k=1)[0]
            
            # Logic: RFSTDTC <= TRTSDT <= TRTEDT
            
            # 1. Informed Consent / Reference Start (RFSTDTC)
            # Randomly within Jan-Mar of start year
            rfstdtc_dt = self.generate_date(base_date, 90)
            
            # 2. Treatment Start (TRTSDT)
            # Must be >= RFSTDTC. Let's say 0-30 days after screening/reference.
            trtsdt_dt = self.generate_date(rfstdtc_dt, 30)
            
            # 3. Treatment End (TRTEDT)
            # Must be >= TRTSDT. Let's say treatment duration is 30-180 days.
            duration = random.randint(30, 180)
            trtedt_dt = trtsdt_dt + timedelta(days=duration)

            anchor = {
                "USUBJID": usubjid,
                "ARM": arm,
                "AGE": age,
                "SEX": sex,
                "RFSTDTC": rfstdtc_dt.strftime("%Y-%m-%d"),
                "TRTSDT": trtsdt_dt.strftime("%Y-%m-%d"),
                "TRTEDT": trtedt_dt.strftime("%Y-%m-%d")
            }
            anchors.append(anchor)
            
        return anchors

    def save_anchors(self, anchors, output_path="output/anchors.json"):
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        with open(output_path, 'w') as f:
            json.dump(anchors, f, indent=2)

if __name__ == "__main__":
    gen = AnchorGenerator()
    anchors = gen.generate_anchors(10)
    print(json.dumps(anchors[:2], indent=2))
    gen.save_anchors(anchors)
