from typing import Dict, Any, Optional, List

class RecommendationService:
    """
    Intelligent Health & Wellness Recommendation Engine.
    Synthesizes Computer Vision facial appearance analysis (dark circles, tiredness,
    acne/blemishes, pigmentation/erythema, hair) and ESP-32 hardware physiological vitals
    to generate evidence-based health findings, resting recommendations, and lifestyle guidance.
    
    DISCLAIMER: Non-prescriptive academic tool. NOT a clinical diagnosis.
    """

    @classmethod
    def generate_recommendations(
        cls,
        visual_features: Dict[str, Any],
        sensor_readings: Optional[Dict[str, Any]],
        hardware_connected: bool = False,
        anomaly_analysis: Optional[Dict[str, Any]] = None,
        baseline_deviation: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Builds a comprehensive health finding report with personalized resting,
        skincare, fatigue, and lifestyle recommendations.
        """
        findings = []
        recommendations = []
        lifestyle_tips = []

        # -------------------------------------------------------------------------
        # 1. Under-Eye Dark Circles & Tiredness Assessment
        # -------------------------------------------------------------------------
        eyes = visual_features.get("eyes", {})
        dark_circles = eyes.get("dark_circles", {})
        tiredness = eyes.get("tiredness", {})

        darkness_pct = dark_circles.get("darkness_index_percentage", 0.0)
        severity = dark_circles.get("severity", "None / Minimal")
        tiredness_score = tiredness.get("tiredness_score", 0.0)

        if darkness_pct >= 10.0 or severity in ("Moderate", "Prominent"):
            findings.append(f"Noticeable under-eye dark circles detected ({severity}, {darkness_pct}% darkness contrast relative to cheek tone).")
            recommendations.append("You need to take rest: Visible infraorbital dark circles indicate physiological fatigue or disrupted sleep cycles. Prioritize 7-8 hours of continuous sleep and elevate your head slightly during rest to prevent fluid accumulation.")
            lifestyle_tips.append("Apply a cool compress or chilled caffeinated eye gel to constrict dilating under-eye capillaries.")
        elif darkness_pct >= 5.0:
            findings.append(f"Mild under-eye shade variation ({darkness_pct}% contrast).")
            recommendations.append("Ensure consistent sleep hygiene and stay adequately hydrated throughout the day.")
        else:
            findings.append("Under-eye periorbital area appears well-rested with nominal pigmentation contrast.")

        if tiredness_score >= 60.0:
            findings.append(f"Visual fatigue index elevated at {tiredness_score}% (eyelid droopiness and eye openness reduction detected).")
            recommendations.append("Immediate eye rest recommended: Practice the 20-20-20 rule (every 20 minutes, look at an object 20 feet away for 20 seconds) to alleviate digital ocular strain.")
            lifestyle_tips.append("Reduce ambient screen glare and avoid smartphone/laptop usage at least 45 minutes before sleep.")
        elif tiredness_score >= 35.0:
            findings.append(f"Mild eye fatigue noted ({tiredness_score}%).")
            lifestyle_tips.append("Take regular micro-breaks from monitor screens and stay hydrated.")

        # -------------------------------------------------------------------------
        # 2. Acne, Blemishes, and Skin Lesion Assessment
        # -------------------------------------------------------------------------
        skin = visual_features.get("skin", {})
        blemishes = skin.get("blemishes", {})
        lesion_count = blemishes.get("visible_lesion_count", 0)
        blemish_severity = blemishes.get("blemish_severity", "Clear / Minimal")
        regions = blemishes.get("candidate_regions", [])

        if lesion_count > 0:
            reg_str = f" in {', '.join(regions)}" if regions else ""
            findings.append(f"Detected {lesion_count} visible candidate acne/blemish lesion(s) ({blemish_severity}){reg_str}.")
            if lesion_count >= 5:
                recommendations.append(f"Blemish care: {blemish_severity} acne-like lesions detected. Cleanse facial skin twice daily with a gentle, non-comedogenic cleanser containing salicylic acid or mild benzoyl peroxide. Avoid picking or manual extraction to prevent post-inflammatory hyperpigmentation.")
            else:
                recommendations.append("Mild blemish spots noted: Maintain gentle cleansing, avoid heavy comedogenic oils, and change pillowcases regularly.")
            lifestyle_tips.append("Support skin microbiome recovery by staying hydrated and limiting excess high-glycemic sugar intake.")
        else:
            findings.append("Facial skin appears clear with minimal visible acne or inflammatory lesions.")

        # -------------------------------------------------------------------------
        # 3. Pigmentation & Redness (Erythema) Assessment
        # -------------------------------------------------------------------------
        pigmentation = skin.get("pigmentation", {})
        erythema_idx = pigmentation.get("erythema_index", 0.0)
        redness_flag = pigmentation.get("redness_flag", False)
        hyper_flag = pigmentation.get("hyperpigmentation_flag", False)
        red_regions = pigmentation.get("redness_regions", [])

        if redness_flag:
            reg_str = f" in {', '.join(red_regions)}" if red_regions else "on facial zones"
            findings.append(f"Localized redness / erythema candidate patches detected (Erythema Index: {erythema_idx}){reg_str}.")
            recommendations.append("Skin barrier caution: Localized erythema observed. Avoid harsh mechanical scrubs, active retinoids, or hot water on inflamed areas. Apply a soothing moisturizer containing ceramides, panthenol, or centella asiatica to repair the epidermal barrier.")
            lifestyle_tips.append("Always apply broad-spectrum mineral sunscreen (SPF 30+) daily to prevent UV-aggravated redness.")
        elif hyper_flag:
            findings.append("Uneven skin tone / localized hyperpigmentation variance observed.")
            recommendations.append("Incorporate gentle topical antioxidants such as Vitamin C or 2-5% Niacinamide to support an even skin tone.")
        else:
            findings.append("Skin tone uniformity is balanced with nominal erythema levels.")

        # -------------------------------------------------------------------------
        # 4. Skin Condition Pattern Screening (SCIN / Fitzpatrick17k taxonomy)
        # -------------------------------------------------------------------------
        cond_pattern = skin.get("condition_pattern", {})
        if cond_pattern.get("psoriasis_pattern_detected", False):
            findings.append("Visual pattern notice: Elevated redness combined with scaly texture variance matching psoriasis-like appearance patterns in benchmark datasets (SCIN / Fitzpatrick17k).")
            recommendations.append("Consultation suggested: For persistent scaly or plaques-like skin patterns, consult a board-certified dermatologist for clinical evaluation.")
        elif cond_pattern.get("dermatitis_pattern_detected", False):
            findings.append("Visual pattern notice: Diffuse erythema with surface texture alteration consistent with dermatitis/eczema-like appearance patterns.")
            recommendations.append("Barrier protection: Minimize fragranced products and environmental allergens. Consult a clinician if itching or scaling persists.")

        # -------------------------------------------------------------------------
        # 5. Hair & Scalp Assessment
        # -------------------------------------------------------------------------
        hair = visual_features.get("hair", {})
        if hair.get("hair_detected", False):
            density = hair.get("hair_density_score", 0.0)
            stability = hair.get("hairline_stability", "Normal")
            gray_pct = hair.get("graying_ratio_percentage", 0.0)
            findings.append(f"Hair & hairline analysis: {stability} (Density score: {density}%, Estimated graying ratio: {gray_pct}%).")
            lifestyle_tips.append("Maintain adequate dietary intake of biotin, zinc, iron, and lean proteins to support follicular keratin synthesis.")

        # -------------------------------------------------------------------------
        # 6. Hardware Sensor Assessment (NO STATIC / FAKE DATA)
        # -------------------------------------------------------------------------
        if not hardware_connected or not sensor_readings or sensor_readings.get("heart_rate") is None:
            hardware_status = "Disconnected / Missing"
            sensor_message = "Hardware physiological sensors (ESP-32) are currently DISCONNECTED or NOT TRANSMITTING. Heart Rate, SpO2, and Body Temperature readings are unavailable. Connect ESP-32 sensor node to activate full multimodal vital sign fusion."
            findings.append("Physiological Vitals: Unavailable (ESP-32 Hardware Disconnected).")
        else:
            hardware_status = "Connected & Active"
            hr = sensor_readings.get("heart_rate")
            spo2 = sensor_readings.get("spo2")
            temp = sensor_readings.get("temperature")

            vitals_summary = f"Heart Rate: {hr} bpm | SpO2: {spo2}% | Temperature: {temp}°C"
            sensor_message = f"Live ESP-32 Telemetry: {vitals_summary}."
            findings.append(f"Physiological Vitals (ESP-32): {vitals_summary}.")

            # Clinical physiology interpretations
            if hr and hr > 100.0:
                recommendations.append(f"Elevated pulse ({hr} bpm): Tachycardia or stress/exercise response detected. Rest in a seated position and practice 4-7-8 diaphragmatic breathing.")
            elif hr and hr < 55.0:
                findings.append(f"Resting pulse is low ({hr} bpm, bradycardic range). Normal in conditioned athletes; consult a physician if accompanied by dizziness.")

            if spo2 and spo2 < 95.0:
                recommendations.append(f"Blood oxygen ({spo2}%) is below nominal resting threshold (95-100%). Sit upright, take deep breaths, verify finger probe placement, and re-check.")

            if temp and temp >= 37.5:
                recommendations.append(f"Elevated body temperature ({temp}°C) indicates low-grade thermal elevation or fever. Stay hydrated, rest, and monitor temperature.")

        # Ensure balanced, actionable lifestyle tips
        if not lifestyle_tips:
            lifestyle_tips = [
                "Aim for 7 to 8 hours of uninterrupted sleep to support cellular regeneration.",
                "Drink 2.0 to 2.5 liters of clean water daily to maintain epidermal and vascular hydration.",
                "Incorporate a 10-minute mindfulness or walking break during intense cognitive work.",
                "Apply daily broad-spectrum SPF 30+ sunscreen on all exposed facial zones."
            ]

        # Primary Executive Summary Sentence
        if darkness_pct >= 10.0 or tiredness_score >= 60.0:
            executive_summary = f"You need to take rest: Noticeable under-eye dark circles ({darkness_pct}%) and visual tiredness ({tiredness_score}%) detected. Sleep recovery and screen breaks are strongly advised."
        elif lesion_count >= 5:
            executive_summary = f"Mild-to-moderate facial blemishes ({lesion_count} visible spots) detected. Skin barrier hydration and gentle cleansing recommended."
        elif redness_flag:
            executive_summary = f"Localized facial redness / erythema observed. Protective barrier care and soothing hydration advised."
        else:
            executive_summary = "Facial appearance and skin metrics show a balanced, nominal state with minimal fatigue."

        return {
            "executive_summary": executive_summary,
            "hardware_status": hardware_status,
            "hardware_connected": hardware_connected,
            "sensor_message": sensor_message,
            "appearance_findings": findings,
            "specific_recommendations": recommendations,
            "general_lifestyle_tips": lifestyle_tips[:5],
            "disclaimer": (
                "IMPORTANT NOTICE: This software is a preliminary screening and appearance analysis tool. "
                "It is NOT a 100% prescribed or clinical diagnostic tool. The visual findings and recommendations "
                "do not constitute medical diagnoses or prescription treatments. Always consult a licensed healthcare professional "
                "or dermatologist for medical evaluation."
            )
        }
