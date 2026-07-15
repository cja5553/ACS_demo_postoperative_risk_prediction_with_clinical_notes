"""
Hand-curated preoperative notes with postoperative outcome labels.

Every note below was written by hand. Every label was assigned by reading that
specific note and deciding what happened to that specific patient. Nothing here
is generated, sampled, or templated.

Outcomes (all binary 0/1):
    DVT        - postoperative deep vein thrombosis
    Pneumonia  - postoperative pneumonia
    AKI        - postoperative acute kidney injury
    Delirium   - postoperative delirium

NOTE ON PREVALENCE: outcome rates here are deliberately higher than real-world
incidence (true postoperative DVT/pneumonia/AKI run ~1-2%). At n=500 a 1% rate
gives ~5 positive cases, which will not train in a single epoch and makes the
demo look broken. Rates are inflated so the model has recoverable signal. These
are NOT epidemiological estimates.
"""

import pandas as pd

# (note, DVT, Pneumonia, AKI, Delirium)
_CURATED_ROWS = [

    # 1. Hip fracture in a demented octogenarian. The delirium archetype:
    #    baseline cognitive impairment + fracture + unfamiliar environment.
    (
        "84F, ASA 3, for right hip hemiarthroplasty after mechanical fall. "
        "Indication: displaced femoral neck fracture. PMH: mild dementia, HTN, "
        "AF on apixaban, last dose 36h ago. Lives in assisted living, ambulates "
        "with walker. Spinal planned.",
        0, 0, 0, 1,
    ),

    # 2. Young, healthy, short laparoscopic case. Nothing happens.
    (
        "24M, ASA 1, for laparoscopic appendectomy. Indication: acute "
        "appendicitis, 12h RLQ pain, WBC 14. No PMH, no home medications. NKDA.",
        0, 0, 0, 0,
    ),

    # 3. Lung resection in a current smoker with severe COPD. Pneumonia is the
    #    expected complication. Cognitively intact, so no delirium.
    (
        "71M, ASA 3, for right upper lobectomy via VATS. Indication: stage IA "
        "NSCLC. PMH: severe COPD, FEV1 48% predicted, current smoker 50 "
        "pack-years, home O2 at night. Tiotropium, albuterol.",
        0, 1, 0, 0,
    ),

    # 4. Septic shock on pressors with baseline CKD. AKI is near-inevitable;
    #    critical illness drives the delirium. Lungs were never the problem.
    (
        "68F, ASA 4E, for emergent exploratory laparotomy. Indication: "
        "perforated diverticulitis with fecal peritonitis, septic shock on "
        "norepinephrine. PMH: CKD stage 3 baseline Cr 1.6, T2DM, HTN. "
        "Lactate 4.2.",
        0, 0, 1, 1,
    ),

    # 5. Redo sternotomy, long pump run, poor EF, CKD, active smoker. Sick
    #    enough to collect three of four honestly.
    (
        "66M, ASA 4, for redo CABG x3 with aortic valve replacement. "
        "Indication: recurrent angina, severe AS, EF 30%. PMH: prior CABG 2011, "
        "prior MI, CKD stage 3, T2DM on insulin, COPD. Current smoker.",
        0, 1, 1, 1,
    ),

    # 6. Elective joint in a healthy middle-aged patient. Regional, fast-track.
    (
        "58F, ASA 2, for left total knee arthroplasty, elective. Indication: "
        "end-stage osteoarthritis. PMH: HTN, BMI 34. Lisinopril. Spinal with "
        "adductor canal block planned.",
        0, 0, 0, 0,
    ),

    # 7. DISCORDANT vs row 1: also in his 80s, but independent, cognitively
    #    sharp, short local case. Age alone does not buy you delirium.
    (
        "82M, ASA 3, for elective open right inguinal hernia repair with mesh. "
        "Indication: symptomatic reducible inguinal hernia. PMH: HTN, BPH. "
        "Independent, still driving, no cognitive complaints. Local with "
        "sedation planned.",
        0, 0, 0, 0,
    ),

    # 8. Pancreatic malignancy, long open abdominal case, deconditioned, low
    #    albumin. Cancer is the VTE driver here.
    (
        "64M, ASA 3, for pancreaticoduodenectomy. Indication: pancreatic head "
        "adenocarcinoma with biliary obstruction, s/p ERCP stent. PMH: T2DM, "
        "15 lb weight loss, deconditioned. Albumin 2.9.",
        1, 0, 0, 0,
    ),

    # 9. Open aortic work: clamp time, contrast, blood loss, baseline CKD, and
    #    a moderate-COPD chest. Renal and pulmonary both give way; his head
    #    does not.
    (
        "73M, ASA 4, for open infrarenal AAA repair. Indication: 6.2 cm "
        "aneurysm with rapid expansion. PMH: HTN, hyperlipidemia, CKD stage 3 "
        "Cr 1.5, moderate COPD, former smoker 40 pack-years.",
        0, 1, 1, 0,
    ),

    # 10. Outpatient endoscopic surveillance. Minimal physiologic insult.
    (
        "61M, ASA 2, for cystoscopy with transurethral resection of bladder "
        "tumor. Indication: recurrent non-muscle-invasive bladder cancer on "
        "surveillance. PMH: HTN, former smoker. Outpatient.",
        0, 0, 0, 0,
    ),

    # 11. Nursing-home nonagenarian-adjacent, emergent, unstable, CKD 4,
    #     demented, obstructed bowel. Aspiration risk on induction.
    (
        "88F, ASA 4E, for emergent laparotomy. Indication: incarcerated ventral "
        "hernia with small bowel obstruction, hemodynamically unstable. PMH: "
        "dementia, AF, CHF EF 35%, CKD stage 4. Admitted from nursing home.",
        0, 1, 1, 1,
    ),

    # 12. Scheduled obstetric case, healthy parturient, spinal.
    (
        "31F, ASA 2, for scheduled repeat cesarean section. Indication: two "
        "prior low transverse cesareans, declined trial of labor. PMH "
        "unremarkable. Spinal planned.",
        0, 0, 0, 0,
    ),

    # 13. DISCORDANT vs row 1: same fracture pattern, same decade of life, but
    #     no baseline cognitive impairment and surgery inside 24h. She sails.
    (
        "76F, ASA 2, for left hip intramedullary nailing. Indication: "
        "intertrochanteric fracture, fall from standing. PMH: osteoporosis, "
        "hypothyroidism. Independent, lives alone, no cognitive issues. To OR "
        "within 24h.",
        0, 0, 0, 0,
    ),

    # 14. DISCORDANT vs row 3: also a lung resection, but never-smoker, normal
    #     PFTs, small wedge. The chest holds.
    (
        "55F, ASA 2, for VATS wedge resection. Indication: solitary pulmonary "
        "nodule, suspected metastasis from prior breast primary. PMH: breast "
        "cancer s/p lumpectomy and radiation 2019. Never smoker. PFTs normal.",
        0, 0, 0, 0,
    ),

    # 15. DISCORDANT on AKI: cardiac surgery, but normal baseline creatinine and
    #     the kidneys are fine. The MCI and prior stroke still cost him his head.
    (
        "79M, ASA 4, for aortic valve replacement, bioprosthetic. Indication: "
        "severe symptomatic AS with syncope. PMH: HTN, mild cognitive "
        "impairment, prior CVA with residual left-sided weakness. Creatinine "
        "0.9, normal renal function.",
        0, 0, 0, 1,
    ),

    # 16. Already dialysis-dependent, so AKI is not on the table. Immobility,
    #     PVD, and active infection drive the clot.
    (
        "67M, ASA 4, for right below-knee amputation. Indication: non-healing "
        "diabetic foot ulcer with osteomyelitis, failed revascularization. PMH: "
        "T2DM with neuropathy, ESRD on hemodialysis, PVD, prior contralateral "
        "transmetatarsal amputation.",
        1, 0, 0, 0,
    ),

    # 17. Long prone multilevel fusion, obese, OSA. Positioning plus
    #     postoperative immobility is the VTE story.
    (
        "59M, ASA 3, for L3-S1 posterior instrumented fusion. Indication: "
        "degenerative spondylolisthesis with neurogenic claudication. PMH: BMI "
        "38, OSA on CPAP, T2DM, HTN. Anticipated prolonged prone positioning.",
        1, 0, 0, 0,
    ),

    # 18. Bread-and-butter elective lap chole.
    (
        "44F, ASA 2, for laparoscopic cholecystectomy. Indication: symptomatic "
        "cholelithiasis with biliary colic. PMH: GERD, BMI 31. Omeprazole.",
        0, 0, 0, 0,
    ),

    # 19. Bedbound post-stroke patient being fed because he keeps aspirating.
    #     The pneumonia is almost the indication. Immobility gives the DVT.
    (
        "85M, ASA 3, for percutaneous endoscopic gastrostomy. Indication: "
        "dysphagia following CVA six weeks ago, failed swallow evaluation, "
        "recurrent aspiration. PMH: prior CVA with right hemiparesis, HTN, AF "
        "on warfarin. Bedbound.",
        1, 1, 0, 1,
    ),

    # 20. Elective neck case, euthyroid, otherwise well.
    (
        "38F, ASA 2, for total thyroidectomy. Indication: multinodular goiter "
        "with compressive symptoms. PMH: hypothyroidism on levothyroxine. "
        "Euthyroid preoperatively.",
        0, 0, 0, 0,
    ),

    # 21. Elective carotid work, cognitively intact, short case.
    (
        "70M, ASA 3, for left carotid endarterectomy. Indication: asymptomatic "
        "80% internal carotid stenosis on duplex. PMH: HTN, hyperlipidemia, "
        "former smoker. Independent, works part-time.",
        0, 0, 0, 0,
    ),

    # 22. Routine TURP. Short, spinal, home next day.
    (
        "74M, ASA 3, for transurethral resection of prostate. Indication: BPH "
        "with urinary retention, failed trial of void. PMH: HTN, "
        "hyperlipidemia. Tamsulosin, finasteride.",
        0, 0, 0, 0,
    ),

    # 23. Anticoagulated fall with a bleed. Head injury plus emergent
    #     craniotomy in his late 70s buys delirium outright.
    (
        "79M, ASA 4E, for emergent craniotomy and evacuation of subdural "
        "hematoma. Indication: acute-on-chronic SDH after fall, GCS 12, "
        "declining. PMH: AF on warfarin, INR 3.1, HTN.",
        1, 0, 0, 1,
    ),

    # 24. Pelvic dissection for colorectal malignancy. Cancer plus pelvic
    #     venous stasis is the VTE combination.
    (
        "68M, ASA 3, for laparoscopic sigmoid colectomy. Indication: sigmoid "
        "adenocarcinoma, T3N1 on staging. PMH: HTN, T2DM, BMI 31. "
        "Metformin, amlodipine.",
        1, 0, 0, 0,
    ),

    # 25. Topical anesthesia, fifteen minutes, no physiologic insult.
    (
        "78F, ASA 2, for right cataract extraction with intraocular lens. "
        "Indication: visual acuity 20/80, impaired night driving. PMH: HTN, "
        "osteoarthritis. Topical anesthesia planned.",
        0, 0, 0, 0,
    ),

    # 26. Ivor Lewis in a smoker. Esophagectomy carries one of the highest
    #     pulmonary complication rates in general surgery.
    (
        "62M, ASA 3, for Ivor Lewis esophagectomy. Indication: distal "
        "esophageal adenocarcinoma after neoadjuvant chemoradiation. PMH: "
        "Barrett esophagus, current smoker 30 pack-years, BMI 27.",
        0, 1, 0, 0,
    ),

    # 27. Bariatric surgery in a young patient. High BMI is a VTE risk factor
    #     but most sleeves are uneventful; hers was.
    (
        "41F, ASA 3, for laparoscopic sleeve gastrectomy. Indication: class III "
        "obesity, BMI 47, failed medical weight management. PMH: OSA on CPAP, "
        "T2DM, HTN. Preop weight loss achieved.",
        1, 0, 0, 0,
    ),

    # 28. Nonagenarian from a nursing home with dementia. Delirium expected;
    #     the aspiration on the ward gave her the pneumonia too.
    (
        "91F, ASA 4, for left hip hemiarthroplasty. Indication: displaced "
        "femoral neck fracture, unwitnessed fall. PMH: advanced dementia, HTN, "
        "CHF. Nursing home resident, wheelchair-bound at baseline.",
        0, 1, 0, 1,
    ),

    # 29. Already dialysis-dependent, so AKI is off the table. Short forearm
    #     case under regional.
    (
        "58M, ASA 4, for left brachiocephalic arteriovenous fistula creation. "
        "Indication: ESRD on hemodialysis via tunneled catheter, needs "
        "permanent access. PMH: ESRD from hypertensive nephrosclerosis, HTN, "
        "CAD.",
        0, 0, 0, 0,
    ),

    # 30. Long open pelvic case with bowel diversion. Cancer plus operative
    #     time plus blood loss; the kidneys took a hit from the ureteral
    #     manipulation and hypotension.
    (
        "71M, ASA 3, for radical cystectomy with ileal conduit. Indication: "
        "muscle-invasive urothelial carcinoma. PMH: former smoker 45 "
        "pack-years, HTN, CKD stage 2. Estimated 5h case.",
        1, 0, 1, 0,
    ),

    # 31. Elective upper extremity arthroplasty, interscalene block, home same
    #     day.
    (
        "66F, ASA 2, for right reverse total shoulder arthroplasty. Indication: "
        "rotator cuff arthropathy with pseudoparalysis. PMH: hypothyroidism, "
        "osteoporosis. Interscalene block planned.",
        0, 0, 0, 0,
    ),

    # 32. Partial nephrectomy with a normal contralateral kidney. Nephron-
    #     sparing, short clamp time; creatinine never moved.
    (
        "59M, ASA 2, for robotic partial nephrectomy. Indication: 3.5 cm "
        "enhancing right renal mass, incidental on imaging. PMH: HTN. Normal "
        "contralateral kidney, creatinine 0.9.",
        0, 0, 0, 0,
    ),

    # 33. Young trauma patient, physiologically robust, recovered fast.
    (
        "27M, ASA 3E, for emergent splenectomy. Indication: grade IV splenic "
        "laceration after motor vehicle collision, failed nonoperative "
        "management. No PMH. Two units transfused.",
        0, 0, 0, 0,
    ),

    # 34. Frail octogenarian, structural heart procedure with contrast load.
    #     Frailty drove the delirium; the dye plus baseline vascular disease
    #     drove the creatinine bump.
    (
        "84F, ASA 4, for transcatheter aortic valve replacement, transfemoral. "
        "Indication: severe symptomatic aortic stenosis, high surgical risk. "
        "PMH: HTN, CKD stage 3, frailty, gait speed <0.5 m/s.",
        0, 0, 1, 1,
    ),

    # 35. Elective benign gynecologic case in a healthy patient.
    (
        "47F, ASA 2, for total laparoscopic hysterectomy. Indication: "
        "symptomatic uterine fibroids with menorrhagia and anemia. PMH: iron "
        "deficiency anemia. Hemoglobin 10.4 preoperatively.",
        0, 0, 0, 0,
    ),

    # 36. Obstructed bowel with a distended stomach; aspiration on induction
    #     despite RSI is the story.
    (
        "73M, ASA 3E, for exploratory laparotomy with adhesiolysis. Indication: "
        "adhesive small bowel obstruction, failed conservative management. PMH: "
        "prior colectomy 2015, prior appendectomy, COPD. NG output 1.8 L.",
        0, 1, 0, 0,
    ),

    # 37. Healthy child, short outpatient ENT case.
    (
        "8M, ASA 1, for tonsillectomy and adenoidectomy. Indication: recurrent "
        "streptococcal tonsillitis, seven episodes in twelve months. No PMH. "
        "Outpatient.",
        0, 0, 0, 0,
    ),

    # 38. Isolated extremity fracture in a healthy adult.
    (
        "34F, ASA 1, for open reduction internal fixation of right ankle. "
        "Indication: bimalleolar fracture after inversion injury. No PMH. "
        "Popliteal block planned.",
        0, 0, 0, 0,
    ),

    # 39. Metastatic disease plus a major hepatectomy. Malignancy is the
    #     dominant VTE driver.
    (
        "61F, ASA 3, for right hepatectomy. Indication: three colorectal liver "
        "metastases, s/p neoadjuvant FOLFOX. PMH: colon cancer s/p right "
        "hemicolectomy 2023, HTN. Future liver remnant adequate.",
        1, 0, 0, 0,
    ),

    # 40. Awake-capable middle-aged patient, elective tumor resection, no
    #     cognitive baseline issues.
    (
        "52M, ASA 2, for left frontal craniotomy for tumor resection. "
        "Indication: enhancing lesion, suspected high-grade glioma. PMH: new-"
        "onset seizure, now on levetiracetam. Otherwise well.",
        0, 0, 0, 0,
    ),

    # 41. Distal revascularization in a diabetic smoker. Uneventful; he keeps
    #     the leg.
    (
        "69M, ASA 4, for right femoral-popliteal bypass with reversed saphenous "
        "vein. Indication: critical limb ischemia with rest pain. PMH: T2DM, "
        "PVD, current smoker, HTN, CAD.",
        0, 0, 0, 0,
    ),

    # 42. DISCORDANT vs row 8: same operation, but this one has COPD. The
    #     chest is what fails, and the cancer still gives him the clot.
    (
        "70M, ASA 4, for pancreaticoduodenectomy. Indication: ampullary "
        "adenocarcinoma with obstructive jaundice. PMH: COPD on home "
        "tiotropium, former smoker 40 pack-years, T2DM, BMI 24. Bilirubin 8.2.",
        1, 1, 0, 0,
    ),

    # 43. Elective breast surgery, healthy, day case.
    (
        "54F, ASA 2, for right mastectomy with sentinel lymph node biopsy. "
        "Indication: invasive ductal carcinoma, ER positive, clinically node "
        "negative. PMH: hypothyroidism.",
        0, 0, 0, 0,
    ),

    # 44. Ruptured aneurysm. Hemorrhagic shock, massive transfusion, clamp
    #     time, prolonged intubation. Almost everything gives way.
    (
        "76M, ASA 5E, for emergent open repair of ruptured abdominal aortic "
        "aneurysm. Indication: contained rupture, hypotensive on arrival. PMH: "
        "HTN, COPD, former smoker. Massive transfusion protocol activated.",
        0, 1, 1, 1,
    ),

    # 45. Cirrhosis is the whole story. Portal hypertension plus perioperative
    #     hypotension in a Child B liver drives the renal injury.
    (
        "58M, ASA 4, for open cholecystectomy. Indication: acute calculous "
        "cholecystitis, failed percutaneous drainage. PMH: alcoholic cirrhosis "
        "Child-Pugh B, ascites, thrombocytopenia, portal hypertension.",
        0, 0, 1, 0,
    ),

    # 46. Revision arthroplasty: long case, extensive dissection, delayed
    #     mobilization.
    (
        "72F, ASA 3, for revision right total hip arthroplasty. Indication: "
        "aseptic loosening of acetabular component, progressive pain. PMH: RA "
        "on methotrexate, osteoporosis, HTN. Anticipated 3h case.",
        1, 0, 0, 0,
    ),

    # 47. Lung transplant. Ischemia-reperfusion, calcineurin inhibitors, and
    #     prolonged ICU stay hit the lungs, kidneys, and head together.
    (
        "58F, ASA 4, for bilateral sequential lung transplant. Indication: "
        "idiopathic pulmonary fibrosis, home O2 4 L, listed 14 months. PMH: "
        "IPF, GERD, pulmonary hypertension. On cardiopulmonary bypass.",
        1, 1, 1, 1,
    ),

    # 48. Focused neck exploration, outpatient.
    (
        "62F, ASA 2, for minimally invasive parathyroidectomy. Indication: "
        "primary hyperparathyroidism with nephrolithiasis, single adenoma "
        "localized on sestamibi. PMH: osteopenia, nephrolithiasis.",
        0, 0, 0, 0,
    ),

    # 49. Emergent but young and physiologically intact; contained perforation,
    #     early source control.
    (
        "55M, ASA 3E, for emergent laparotomy with Graham patch repair. "
        "Indication: perforated duodenal ulcer with free air, peritonitis. PMH: "
        "chronic NSAID use for back pain, tobacco use. Presented within 6h.",
        0, 0, 0, 0,
    ),

    # 50. Short decompression, prone, home in two days.
    (
        "68M, ASA 3, for L4-L5 laminectomy. Indication: lumbar spinal stenosis "
        "with neurogenic claudication, failed conservative management. PMH: "
        "HTN, hyperlipidemia, BMI 29.",
        0, 0, 0, 0,
    ),

    # 51. Planned second-stage restoration in a recovered patient.
    (
        "60F, ASA 2, for Hartmann reversal with colorectal anastomosis. "
        "Indication: restoration of continuity nine months after emergent "
        "sigmoidectomy for perforated diverticulitis. PMH: HTN, "
        "diverticulosis.",
        0, 0, 0, 0,
    ),

    # 52. Losing an entire lung in a heavy smoker. The remaining lung cannot
    #     absorb the insult.
    (
        "65M, ASA 4, for right pneumonectomy. Indication: central non-small "
        "cell lung carcinoma involving the main bronchus. PMH: COPD, FEV1 55% "
        "predicted, current smoker 45 pack-years, CAD.",
        0, 1, 0, 0,
    ),

    # 53. DISCORDANT: NPH with gait and cognitive complaints, but the shunt is
    #     a short case and she came through clear.
    (
        "71F, ASA 3, for ventriculoperitoneal shunt placement. Indication: "
        "normal pressure hydrocephalus, gait apraxia improved after high-volume "
        "lumbar puncture. PMH: HTN, mild memory complaints.",
        0, 0, 0, 0,
    ),

    # 54. Elective foregut case in a healthy patient.
    (
        "49M, ASA 2, for laparoscopic Nissen fundoplication. Indication: "
        "refractory GERD despite maximal PPI, positive pH study. PMH: GERD, "
        "hiatal hernia, BMI 30.",
        0, 0, 0, 0,
    ),

    # 55. Heart transplant. Bypass, inotropes, induction immunosuppression,
    #     long ICU course.
    (
        "54M, ASA 4, for orthotopic heart transplant. Indication: end-stage "
        "nonischemic cardiomyopathy, EF 15%, bridged on LVAD 8 months. PMH: "
        "dilated cardiomyopathy, ICD, CKD stage 3.",
        1, 1, 1, 1,
    ),

    # 56. Robotic, minimal blood loss, discharged next morning.
    (
        "63M, ASA 2, for robotic-assisted radical prostatectomy. Indication: "
        "prostate adenocarcinoma, Gleason 3+4, clinically localized. PMH: HTN. "
        "Continent and potent preoperatively.",
        0, 0, 0, 0,
    ),

    # 57. Minor anorectal case, healthy adult.
    (
        "43M, ASA 1, for excisional hemorrhoidectomy. Indication: grade III "
        "internal hemorrhoids with recurrent bleeding, failed banding. No PMH.",
        0, 0, 0, 0,
    ),

    # 58. Urgent obstetric case in a healthy parturient; delivered in minutes.
    (
        "29F, ASA 2E, for emergent cesarean section. Indication: category III "
        "fetal heart tracing, prolonged deceleration. PMH: gestational diabetes "
        "diet-controlled. General anesthesia, decision-to-incision 9 min.",
        0, 0, 0, 0,
    ),

    # 59. DISCORDANT: 81 and a fracture, but short percutaneous case under
    #     sedation and she is cognitively intact.
    (
        "81F, ASA 3, for L1 kyphoplasty. Indication: osteoporotic vertebral "
        "compression fracture with intractable pain. PMH: osteoporosis, HTN, "
        "hypothyroidism. Independent, no cognitive complaints.",
        0, 0, 0, 0,
    ),

    # 60. Large open abdominal wall reconstruction in an obese patient. Long
    #     case, tight closure, but the chest held.
    (
        "62M, ASA 3, for open ventral hernia repair with component separation "
        "and mesh. Indication: recurrent incisional hernia, 14 cm defect. PMH: "
        "BMI 42, T2DM, OSA, three prior abdominal operations.",
        0, 0, 0, 0,
    ),

    # 61. Deceased donor graft with long cold ischemia. Delayed graft function
    #     is exactly what the AKI label captures here.
    (
        "52M, ASA 4, for deceased donor kidney transplant. Indication: ESRD on "
        "hemodialysis 6 years, diabetic nephropathy. PMH: T2DM, HTN, CAD s/p "
        "PCI. Cold ischemia time 22h, extended criteria donor.",
        0, 0, 1, 0,
    ),

    # 62. Elective primary hip in a fit patient, spinal, mobilized same day.
    (
        "64M, ASA 2, for right total hip arthroplasty, primary. Indication: "
        "end-stage osteoarthritis with night pain. PMH: HTN, hyperlipidemia. "
        "Independent, walks 2 miles daily.",
        0, 0, 0, 0,
    ),

    # 63. Myasthenia plus a sternotomy: bulbar weakness and impaired cough are
    #     the setup for postoperative pulmonary trouble.
    (
        "44F, ASA 3, for transsternal thymectomy. Indication: generalized "
        "myasthenia gravis, acetylcholine receptor antibody positive, thymic "
        "hyperplasia. PMH: myasthenia on pyridostigmine and prednisone.",
        0, 1, 0, 0,
    ),

    # 64. Large burn, repeated trips to the OR, inhalation component.
    (
        "37M, ASA 4, for excisional debridement and split-thickness skin "
        "grafting, 28% TBSA. Indication: flame burn to torso and bilateral arms, "
        "day 4. PMH: none. Intubated for inhalation injury.",
        1, 1, 0, 0,
    ),

    # 65. Eight hours of microsurgery with the patient immobile on the table,
    #     plus an active malignancy.
    (
        "51F, ASA 3, for bilateral mastectomy with immediate DIEP flap "
        "reconstruction. Indication: invasive lobular carcinoma, BRCA2 "
        "positive. PMH: BMI 33, HTN. Anticipated 8h case.",
        1, 0, 0, 0,
    ),

    # 66. Catecholamine-secreting tumor. Blocked preoperatively; the case went
    #     smoothly and she was extubated on the table.
    (
        "46F, ASA 3, for laparoscopic right adrenalectomy. Indication: "
        "pheochromocytoma, 4 cm, biochemically confirmed. PMH: paroxysmal "
        "hypertension. Alpha-blocked on phenoxybenzamine for 14 days.",
        0, 0, 0, 0,
    ),

    # 67. Dead bowel with lactic acidosis and vasopressors. The kidneys go
    #     first; the ICU course takes his head.
    (
        "77M, ASA 5E, for emergent laparotomy with small bowel resection. "
        "Indication: acute mesenteric ischemia, SMA occlusion, lactate 7.1, on "
        "vasopressors. PMH: AF not anticoagulated, CHF, CKD stage 3.",
        0, 0, 1, 1,
    ),

    # 68. Airway malignancy in a heavy smoker; permanent stoma and impaired
    #     pulmonary toilet.
    (
        "66M, ASA 3, for total laryngectomy with bilateral neck dissection. "
        "Indication: T4 laryngeal squamous cell carcinoma. PMH: current smoker "
        "50 pack-years, heavy alcohol use, COPD, malnutrition.",
        0, 1, 0, 0,
    ),

    # 69. Young athlete, arthroscopic, regional block, home same day.
    (
        "23M, ASA 1, for arthroscopic anterior cruciate ligament reconstruction "
        "with hamstring autograft. Indication: complete ACL tear, recurrent "
        "instability. No PMH.",
        0, 0, 0, 0,
    ),

    # 70. Second-trimester lap chole. Healthy, brief, uncomplicated.
    (
        "28F, ASA 2, for laparoscopic cholecystectomy at 19 weeks gestation. "
        "Indication: recurrent biliary colic with gallstone pancreatitis, "
        "resolved. PMH: pregnancy otherwise uncomplicated.",
        0, 0, 0, 0,
    ),

    # 71. Steroid-dependent colitis in a young patient. Sick colon, but a
    #     resilient host.
    (
        "31F, ASA 3, for total abdominal colectomy with end ileostomy. "
        "Indication: medically refractory ulcerative colitis, steroid "
        "dependent. PMH: UC 9 years, on prednisone 40 mg and infliximab.",
        0, 0, 0, 0,
    ),

    # 72. Already intubated and failing to wean. Ventilator-associated
    #     pneumonia is the expected event.
    (
        "69M, ASA 4, for percutaneous tracheostomy. Indication: failure to wean "
        "after 12 days of mechanical ventilation following COPD exacerbation. "
        "PMH: severe COPD, cor pulmonale, former smoker.",
        0, 1, 0, 0,
    ),

    # 73. Necrotizing infection with septic physiology in a diabetic. Source
    #     control plus pressors; the kidneys pay.
    (
        "56M, ASA 4E, for emergent radical debridement of perineum and left "
        "thigh. Indication: Fournier gangrene with septic shock. PMH: T2DM "
        "poorly controlled HbA1c 11.2, obesity, on norepinephrine.",
        0, 0, 1, 0,
    ),

    # 74. Ovarian debulking: long, bloody, and driven by advanced malignancy.
    (
        "63F, ASA 3, for exploratory laparotomy with tumor debulking, total "
        "abdominal hysterectomy, bilateral salpingo-oophorectomy, omentectomy. "
        "Indication: stage IIIC ovarian carcinoma with carcinomatosis.",
        1, 0, 0, 0,
    ),

    # 75. Arthroscopic shoulder work, healthy, sling and home.
    (
        "48M, ASA 2, for arthroscopic rotator cuff repair. Indication: "
        "full-thickness supraspinatus tear after fall, failed physiotherapy. "
        "PMH: HTN. Interscalene block planned.",
        0, 0, 0, 0,
    ),

    # 76. Aortic root replacement with a long pump run and circulatory arrest;
    #     renal and neurologic consequences follow.
    (
        "58M, ASA 4, for Bentall procedure with composite valve graft. "
        "Indication: aortic root aneurysm 5.6 cm with severe aortic "
        "insufficiency, bicuspid valve. PMH: Marfan syndrome, HTN.",
        0, 0, 1, 1,
    ),

    # 77. Elective mitral repair in a fit patient; short pump run, fast track.
    (
        "56F, ASA 3, for minimally invasive mitral valve repair. Indication: "
        "severe mitral regurgitation from posterior leaflet prolapse, "
        "asymptomatic with LV dilation. PMH: otherwise well. EF 60%.",
        0, 0, 0, 0,
    ),

    # 78. Twenty minutes, local, no sedation.
    (
        "52F, ASA 2, for right carpal tunnel release. Indication: carpal tunnel "
        "syndrome with thenar atrophy, confirmed on EMG. PMH: hypothyroidism, "
        "T2DM. Local anesthesia, wide awake.",
        0, 0, 0, 0,
    ),

    # 79. Six-week-old infant, short case, home the next day.
    (
        "6-week-old male, ASA 2, for laparoscopic pyloromyotomy. Indication: "
        "hypertrophic pyloric stenosis with projectile vomiting. Electrolytes "
        "corrected preoperatively, chloride 102.",
        0, 0, 0, 0,
    ),

    # 80. Another hip fracture in an elderly man, this one with alcohol history
    #     and a delayed presentation. Withdrawal compounds the delirium.
    (
        "87M, ASA 4, for right hip intramedullary nailing. Indication: "
        "subtrochanteric fracture, found down at home, unclear duration. PMH: "
        "chronic alcohol use, malnutrition, HTN, prior falls. Rhabdomyolysis on "
        "admission, CK 8400.",
        0, 0, 1, 1,
    ),

    # 81. DISCORDANT vs row 30: also a urologic malignancy in a smoker, but a
    #     clean flank nephrectomy with a normal contralateral kidney.
    (
        "60M, ASA 3, for left radical nephrectomy. Indication: 8 cm renal cell "
        "carcinoma, no venous involvement. PMH: HTN, former smoker 20 "
        "pack-years, BMI 28. Creatinine 1.0.",
        0, 0, 0, 0,
    ),

    # 82. DISCORDANT vs rows 9 and 44: same aneurysm, endovascular approach.
    #     No clamp, no laparotomy, but a big contrast load on a stiff kidney.
    (
        "75M, ASA 4, for endovascular aneurysm repair. Indication: 5.8 cm "
        "infrarenal AAA, favorable anatomy. PMH: CKD stage 3 Cr 1.7, COPD, HTN, "
        "former smoker. Contrast volume 140 mL.",
        0, 0, 1, 0,
    ),

    # 83. Short endoscopic stone case, spinal, home same day.
    (
        "39F, ASA 2, for ureteroscopy with laser lithotripsy and stent "
        "placement. Indication: obstructing 8 mm left distal ureteral calculus. "
        "PMH: recurrent nephrolithiasis. Afebrile, no infection.",
        0, 0, 0, 0,
    ),

    # 84. Young Crohn's patient on biologics; ileocecal resection, uneventful.
    (
        "29M, ASA 2, for laparoscopic ileocecal resection. Indication: "
        "stricturing Crohn disease with obstructive symptoms, failed medical "
        "therapy. PMH: Crohn disease 11 years, on adalimumab.",
        0, 0, 0, 0,
    ),

    # 85. Ruptured aneurysm with subarachnoid blood. Vasospasm protocol,
    #     prolonged ICU, and blood in the cisterns; the head does not come back
    #     clean.
    (
        "54F, ASA 4E, for emergent craniotomy with aneurysm clipping. "
        "Indication: ruptured anterior communicating artery aneurysm, Hunt-Hess "
        "3, Fisher 4. PMH: HTN, current smoker.",
        1, 1, 0, 1,
    ),

    # 86. Elective otologic case in a healthy adult.
    (
        "45F, ASA 2, for right cochlear implantation. Indication: bilateral "
        "profound sensorineural hearing loss, no benefit from hearing aids. "
        "PMH: otosclerosis. Otherwise well.",
        0, 0, 0, 0,
    ),

    # 87. Large panniculectomy after massive weight loss. Long case, big
    #     dissection, but a young and mobile patient.
    (
        "43F, ASA 3, for panniculectomy. Indication: symptomatic pannus with "
        "recurrent intertrigo after 120 lb weight loss post-bypass. PMH: prior "
        "Roux-en-Y 2021, resolved T2DM, BMI now 31.",
        0, 0, 0, 0,
    ),

    # 88. Liver transplant. Reperfusion, massive transfusion, calcineurin
    #     inhibitors, and encephalopathy that predates the operation.
    (
        "59M, ASA 4, for orthotopic liver transplant. Indication: "
        "decompensated cirrhosis, MELD 31, refractory ascites and hepatic "
        "encephalopathy. PMH: alcoholic cirrhosis, hepatorenal syndrome, "
        "esophageal varices.",
        0, 1, 1, 1,
    ),

    # 89. Perforated appendicitis in an older patient with a delayed
    #     presentation. Contained abscess, but she was dry on arrival.
    (
        "72F, ASA 3E, for open appendectomy with abscess drainage. Indication: "
        "perforated appendicitis with periappendiceal abscess, 5 days of "
        "symptoms. PMH: HTN, CKD stage 3, T2DM. Creatinine 2.1 on admission.",
        0, 0, 1, 0,
    ),

    # 90. Emergent groin hernia with compromised bowel in a frail woman.
    (
        "83F, ASA 4E, for emergent right femoral hernia repair with small bowel "
        "resection. Indication: incarcerated femoral hernia with ischemic "
        "segment. PMH: HTN, osteoporosis, mild cognitive impairment. Lives "
        "alone.",
        0, 0, 0, 1,
    ),

    # 91. Wide local excision and node biopsy, healthy patient, outpatient.
    (
        "50M, ASA 2, for wide local excision of left back melanoma with "
        "sentinel lymph node biopsy. Indication: 2.1 mm Breslow depth melanoma, "
        "clinically node negative. PMH: none.",
        0, 0, 0, 0,
    ),

    # 92. Elective ankle arthroplasty, regional, weight-bearing restricted but
    #     she is otherwise well.
    (
        "61F, ASA 2, for right total ankle arthroplasty. Indication: "
        "post-traumatic ankle arthritis 12 years after pilon fracture. PMH: "
        "hypothyroidism, BMI 27.",
        0, 0, 0, 0,
    ),

    # 93. Brief diagnostic case under general anesthesia.
    (
        "58M, ASA 3, for mediastinoscopy with lymph node sampling. Indication: "
        "mediastinal lymphadenopathy on staging CT, suspected lymphoma. PMH: "
        "HTN, former smoker. Awaiting tissue diagnosis.",
        0, 0, 0, 0,
    ),

    # 94. Risk-reducing surgery in a healthy carrier.
    (
        "42F, ASA 1, for laparoscopic bilateral salpingo-oophorectomy. "
        "Indication: risk-reducing surgery, BRCA1 carrier, family history of "
        "ovarian cancer. PMH: none. Premenopausal.",
        0, 0, 0, 0,
    ),

    # 95. Amputation above the knee in a diabetic with dead tissue and no
    #     mobility left. Immobility drives the clot.
    (
        "74M, ASA 4, for left above-knee amputation. Indication: failed BKA "
        "stump with wet gangrene, unreconstructable disease. PMH: T2DM, PVD, "
        "CKD stage 4, CHF, prior contralateral AKA. Wheelchair-bound.",
        1, 0, 0, 0,
    ),

    # 96. Elective implantable device, sedation, short case.
    (
        "47F, ASA 3, for spinal cord stimulator implantation. Indication: "
        "failed back surgery syndrome, successful trial. PMH: chronic pain on "
        "long-acting opioids, depression, prior L4-L5 fusion.",
        0, 0, 0, 0,
    ),

    # 97. Total gastrectomy for cancer: upper abdominal incision, diaphragmatic
    #     splinting, malnourished host.
    (
        "67M, ASA 3, for total gastrectomy with Roux-en-Y reconstruction and D2 "
        "lymphadenectomy. Indication: gastric adenocarcinoma. PMH: 20 lb weight "
        "loss, albumin 2.8, former smoker, HTN.",
        1, 1, 0, 0,
    ),

    # 98. Short elective nasal case, healthy adult.
    (
        "33M, ASA 1, for septoplasty with inferior turbinate reduction. "
        "Indication: nasal obstruction from deviated septum, failed medical "
        "management. No PMH. Outpatient.",
        0, 0, 0, 0,
    ),

    # 99. Reopening a fresh sternotomy for tamponade. Unstable, but the problem
    #     is mechanical and the fix is immediate.
    (
        "70M, ASA 4E, for emergent resternotomy for cardiac tamponade. "
        "Indication: hypotension and rising filling pressures 6h after CABG. "
        "PMH: CAD, T2DM, HTN, CABG earlier today.",
        0, 0, 1, 1,
    ),

    # 100. Elective hernia in a young healthy man, local plus sedation.
    (
        "36M, ASA 1, for laparoscopic bilateral inguinal hernia repair with "
        "mesh. Indication: bilateral symptomatic inguinal hernias, manual "
        "labourer. No PMH.",
        0, 0, 0, 0,
    ),

    # 101. Both knees at once: double the tourniquet time, double the blood
    #      loss, and immobility afterwards.
    (
        "67F, ASA 3, for bilateral simultaneous total knee arthroplasty. "
        "Indication: bilateral end-stage osteoarthritis, wheelchair-dependent "
        "from pain. PMH: BMI 39, T2DM, HTN, varicose veins.",
        1, 0, 0, 0,
    ),

    # 102. Elderly colon cancer resection. Cancer plus open abdomen plus age.
    (
        "80M, ASA 3, for open right hemicolectomy. Indication: obstructing "
        "cecal adenocarcinoma with iron deficiency anemia. PMH: HTN, AF on "
        "rivaroxaban, prior TIA. Hemoglobin 8.1 preoperatively.",
        1, 0, 0, 1,
    ),

    # 103. DISCORDANT vs row 5: isolated first-time CABG, normal kidneys,
    #      preserved EF, non-smoker. Straightforward pump run, extubated early.
    (
        "61M, ASA 3, for coronary artery bypass grafting x3, elective. "
        "Indication: three-vessel coronary disease, EF 55%, stable angina. PMH: "
        "hyperlipidemia, HTN, former smoker quit 15 years ago. Creatinine 0.9.",
        0, 0, 0, 0,
    ),

    # 104. Type A dissection: circulatory arrest, malperfusion, and a long
    #      bypass run. The kidneys and the brain both take it.
    (
        "63M, ASA 5E, for emergent ascending aortic replacement with "
        "hemiarch. Indication: acute type A aortic dissection with pericardial "
        "effusion. PMH: uncontrolled HTN. Deep hypothermic circulatory arrest "
        "38 min.",
        0, 1, 1, 1,
    ),

    # 105. High-energy fracture in a fit middle-aged man. Nothing about him is
    #      frail.
    (
        "55M, ASA 2, for open reduction internal fixation of right acetabulum. "
        "Indication: displaced posterior wall fracture after motorcycle "
        "collision. PMH: none. Isolated injury.",
        0, 0, 0, 0,
    ),

    # 106. Bypass in a superobese patient with OSA. Uneventful; ambulating the
    #      same evening.
    (
        "38F, ASA 3, for laparoscopic Roux-en-Y gastric bypass. Indication: "
        "class III obesity BMI 51 with T2DM and OSA. PMH: OSA on CPAP, T2DM on "
        "metformin, HTN, GERD.",
        1, 0, 0, 0,
    ),

    # 107. Endometrial staging in an obese patient; robotic, short, home the
    #      next day. Malignancy still counts for the clot.
    (
        "66F, ASA 3, for robotic hysterectomy with bilateral "
        "salpingo-oophorectomy and pelvic lymphadenectomy. Indication: grade 2 "
        "endometrioid adenocarcinoma. PMH: BMI 41, T2DM, HTN.",
        1, 0, 0, 0,
    ),

    # 108. Elective superficial parotid with nerve monitoring; short and clean.
    (
        "57M, ASA 2, for superficial parotidectomy with facial nerve "
        "monitoring. Indication: 3 cm pleomorphic adenoma, enlarging. PMH: "
        "HTN. Otherwise well.",
        0, 0, 0, 0,
    ),

    # 109. Flail segment with pulmonary contusion. The chest wall was the
    #      indication and the lung was already injured.
    (
        "62M, ASA 3E, for surgical stabilization of rib fractures, right chest. "
        "Indication: flail chest with six segmental fractures and pulmonary "
        "contusion after fall from ladder. PMH: COPD, current smoker.",
        0, 1, 0, 0,
    ),

    # 110. Penetrating trauma in a young man. Resuscitated, damage control,
    #      recovered.
    (
        "24M, ASA 4E, for emergent laparotomy for gunshot wound. Indication: "
        "abdominal GSW with hemoperitoneum, hypotensive in trauma bay. No PMH. "
        "Liver packing and small bowel repair, 6 units transfused.",
        0, 0, 0, 0,
    ),

    # 111. Minor day case in a healthy young adult.
    (
        "21M, ASA 1, for excision of pilonidal sinus with primary closure. "
        "Indication: recurrent pilonidal abscess, three prior drainages. No "
        "PMH.",
        0, 0, 0, 0,
    ),

    # 112. Empyema in a debilitated patient; the infection is already in his
    #      chest before the operation starts.
    (
        "59M, ASA 4, for VATS decortication. Indication: stage III empyema with "
        "trapped lung after failed chest tube drainage. PMH: alcohol use "
        "disorder, malnutrition, prior aspiration events, COPD.",
        1, 1, 0, 1,
    ),

    # 113. Frail elderly cholecystectomy. She is sick but the operation is
    #      quick and she stays clear-headed.
    (
        "85F, ASA 4, for laparoscopic cholecystectomy. Indication: acute "
        "cholecystitis, tokyo grade II, failed antibiotics. PMH: CHF EF 40%, "
        "AF, CKD stage 3, HTN. Lives independently with family nearby.",
        0, 0, 1, 0,
    ),

    # 114. Access surgery in a dialysis patient; forearm, regional, quick.
    (
        "64F, ASA 4, for left forearm arteriovenous graft placement. "
        "Indication: ESRD on hemodialysis, exhausted native vein options. PMH: "
        "ESRD from diabetic nephropathy, T2DM, PVD, CAD.",
        0, 0, 0, 0,
    ),

    # 115. Upper tract urothelial cancer; loses a kidney with a stiff one left
    #      behind.
    (
        "73M, ASA 3, for laparoscopic right nephroureterectomy with bladder "
        "cuff excision. Indication: high-grade upper tract urothelial "
        "carcinoma. PMH: CKD stage 3 Cr 1.6, HTN, former smoker 35 pack-years.",
        0, 0, 1, 0,
    ),

    # 116. Elective prolapse repair in a healthy postmenopausal patient.
    (
        "59F, ASA 2, for laparoscopic sacrocolpopexy. Indication: stage III "
        "vaginal vault prolapse after prior hysterectomy. PMH: HTN, prior "
        "vaginal hysterectomy 2016.",
        0, 0, 0, 0,
    ),

    # 117. Sigmoid volvulus in a nursing-home patient with dementia. Detorsion
    #      and resection; the environment change does the rest.
    (
        "86M, ASA 4E, for emergent sigmoid colectomy with end colostomy. "
        "Indication: sigmoid volvulus with failed endoscopic detorsion, "
        "recurrent. PMH: dementia, Parkinson disease, chronic constipation. "
        "Nursing home resident.",
        0, 1, 0, 1,
    ),

    # 118. Arthroplasty in a dialysis-dependent patient. Already anuric, so AKI
    #      is not available to him.
    (
        "62M, ASA 4, for right total hip arthroplasty. Indication: avascular "
        "necrosis of femoral head, end-stage. PMH: ESRD on hemodialysis, "
        "long-term steroid use for lupus, HTN, anemia.",
        1, 0, 0, 0,
    ),

    # 119. Single-level microdiscectomy in a working adult.
    (
        "41F, ASA 2, for L5-S1 microdiscectomy. Indication: herniated disc with "
        "refractory radiculopathy and foot drop. PMH: none. Symptoms 10 weeks.",
        0, 0, 0, 0,
    ),

    # 120. Airway resection with an anastomosis and a chin stitch; pulmonary
    #      toilet is compromised by design.
    (
        "48F, ASA 3, for tracheal resection with end-to-end anastomosis. "
        "Indication: post-intubation tracheal stenosis, 3 cm segment. PMH: "
        "prior prolonged intubation for ARDS 2024, obesity BMI 35, asthma.",
        0, 1, 0, 0,
    ),

    # 121. Elderly patient with a hypopharyngeal pouch; the regurgitation is
    #      the indication and the aspiration follows it into hospital.
    (
        "78M, ASA 3, for open Zenker diverticulectomy with cricopharyngeal "
        "myotomy. Indication: symptomatic pharyngeal pouch with regurgitation "
        "and weight loss. PMH: HTN, prior aspiration pneumonia, GERD.",
        0, 1, 0, 0,
    ),

    # 122. Young patient, endoscopic myotomy, home the next day.
    (
        "35F, ASA 2, for peroral endoscopic myotomy. Indication: type II "
        "achalasia confirmed on manometry, dysphagia to solids and liquids. "
        "PMH: achalasia, 12 lb weight loss.",
        0, 0, 0, 0,
    ),

    # 123. Prophylactic surgery with implants in a healthy young carrier.
    (
        "39F, ASA 1, for bilateral prophylactic mastectomy with tissue expander "
        "placement. Indication: BRCA1 carrier, mother and sister with breast "
        "cancer. PMH: none.",
        0, 0, 0, 0,
    ),

    # 124. Obstructed infected kidney in a diabetic. Decompression is the fix,
    #      but she arrives septic with a stiff kidney behind the stone.
    (
        "68F, ASA 4E, for emergent cystoscopy with left ureteral stent "
        "placement. Indication: obstructing ureteral calculus with pyonephrosis "
        "and urosepsis, febrile to 39.4, hypotensive. PMH: T2DM, HTN, recurrent "
        "UTI.",
        0, 0, 1, 0,
    ),

    # 125. Colectomy in a demented elderly patient. Longer stay, more lines,
    #      more nights away from home.
    (
        "84M, ASA 4, for open low anterior resection with diverting ileostomy. "
        "Indication: rectal adenocarcinoma after neoadjuvant chemoradiation. "
        "PMH: moderate dementia, HTN, CAD, CKD stage 3. Assisted living.",
        1, 0, 0, 1,
    ),

    # 126. DISCORDANT vs row 40: elderly this time, but a small convexity
    #      meningioma and a clean course.
    (
        "76F, ASA 3, for right parietal craniotomy for meningioma resection. "
        "Indication: 3 cm convexity meningioma with progressive headache. PMH: "
        "HTN, hypothyroidism. Independent, MMSE 29.",
        0, 0, 0, 0,
    ),

    # 127. DISCORDANT vs row 22: same operation, ninety years old, and he still
    #      goes home clear.
    (
        "90M, ASA 3, for transurethral resection of prostate. Indication: "
        "refractory urinary retention with indwelling catheter. PMH: HTN, "
        "osteoarthritis. Independent, sharp, plays bridge weekly. Spinal "
        "planned.",
        0, 0, 0, 0,
    ),

    # 128. Radical hysterectomy: pelvic sidewall dissection with malignancy.
    (
        "44F, ASA 2, for radical hysterectomy with pelvic lymphadenectomy. "
        "Indication: stage IB2 cervical squamous cell carcinoma. PMH: former "
        "smoker. Anticipated 4h open case.",
        1, 0, 0, 0,
    ),

    # 129. Valve replacement in an elderly woman with a dilated atrium and a
    #      long pump run.
    (
        "81F, ASA 4, for mitral valve replacement with bioprosthesis and left "
        "atrial appendage ligation. Indication: severe mitral stenosis with "
        "pulmonary hypertension. PMH: rheumatic heart disease, AF, CKD stage 3, "
        "frailty.",
        0, 0, 1, 1,
    ),

    # 130. Long fusion for adult deformity: hours prone, big blood loss,
    #      staged closure.
    (
        "68F, ASA 3, for T10-pelvis posterior instrumented fusion with "
        "osteotomies. Indication: adult degenerative scoliosis with sagittal "
        "imbalance, unable to stand upright. PMH: osteoporosis, HTN, BMI 30. "
        "Anticipated 7h case, EBL 1500 mL.",
        1, 0, 0, 1,
    ),

    # 131. Distal pancreatectomy in a fit patient; laparoscopic, spleen
    #      preserved.
    (
        "54F, ASA 2, for laparoscopic distal pancreatectomy with splenic "
        "preservation. Indication: mucinous cystic neoplasm of the pancreatic "
        "tail, 4 cm, enlarging. PMH: hypothyroidism.",
        0, 0, 0, 0,
    ),

    # 132. Palliative diversion in a patient with carcinomatosis. Advanced
    #      cancer and near-total immobility.
    (
        "70F, ASA 4, for laparoscopic loop ileostomy creation. Indication: "
        "malignant bowel obstruction from peritoneal carcinomatosis, palliative "
        "intent. PMH: stage IV ovarian cancer on chemotherapy, cachexia, "
        "recurrent ascites.",
        1, 0, 0, 0,
    ),

    # 133. Small ischemic digit in a diabetic; local, ten minutes.
    (
        "63M, ASA 3, for right second toe amputation. Indication: dry gangrene "
        "of the toe with adequate perfusion proximally. PMH: T2DM with "
        "neuropathy, HTN, palpable pedal pulses. Local anesthesia.",
        0, 0, 0, 0,
    ),

    # 134. Adult congenital repair, short pump run, young patient.
    (
        "33F, ASA 2, for surgical closure of secundum atrial septal defect. "
        "Indication: hemodynamically significant shunt with right ventricular "
        "dilation, unsuitable for device closure. PMH: ASD, otherwise well.",
        0, 0, 0, 0,
    ),

    # 135. Elective sinus surgery, healthy adult, outpatient.
    (
        "46M, ASA 2, for bilateral functional endoscopic sinus surgery. "
        "Indication: chronic rhinosinusitis with nasal polyposis, failed "
        "maximal medical therapy. PMH: asthma, aspirin sensitivity.",
        0, 0, 0, 0,
    ),

    # 136. Small elective hernia, healthy, day case.
    (
        "45F, ASA 2, for open umbilical hernia repair with mesh. Indication: "
        "symptomatic umbilical hernia, 2 cm defect. PMH: two prior pregnancies, "
        "BMI 28.",
        0, 0, 0, 0,
    ),

    # 137. Open abdomen after resuscitation. Vented, on pressors, kidneys
    #      already squeezed by the intra-abdominal pressure.
    (
        "51M, ASA 5E, for decompressive laparotomy with temporary abdominal "
        "closure. Indication: abdominal compartment syndrome after massive "
        "resuscitation for necrotizing pancreatitis. Bladder pressure 28 mmHg, "
        "anuric, on norepinephrine.",
        1, 1, 1, 1,
    ),

    # 138. Elderly hip fracture on dialysis. Immobility drives the clot; the
    #      kidneys are already gone.
    (
        "79M, ASA 4, for left hip hemiarthroplasty. Indication: displaced "
        "femoral neck fracture after fall during dialysis session. PMH: ESRD on "
        "hemodialysis, T2DM, PVD, prior MI. Uses walker at baseline.",
        1, 0, 0, 1,
    ),

    # 139. Elective thoracic case in a well-preserved patient; nonsmoker,
    #      normal PFTs, small resection.
    (
        "50M, ASA 2, for VATS resection of anterior mediastinal mass. "
        "Indication: 4 cm thymic cyst, symptomatic. PMH: none. Never smoker, "
        "PFTs normal.",
        0, 0, 0, 0,
    ),

    # 140. Emergent amputation in a septic diabetic with a dead foot.
    (
        "71F, ASA 4E, for emergent left below-knee amputation. Indication: wet "
        "gangrene of the forefoot with ascending cellulitis and sepsis. PMH: "
        "T2DM HbA1c 10.8, PVD, CKD stage 3, CHF. Febrile, WBC 22.",
        1, 0, 1, 0,
    ),

    # 141. Destination therapy device in end-stage heart failure. Low output
    #      state, bypass, and right heart failure afterwards.
    (
        "60M, ASA 4, for HeartMate 3 left ventricular assist device "
        "implantation, destination therapy. Indication: NYHA IV nonischemic "
        "cardiomyopathy, EF 12%, inotrope-dependent. PMH: CKD stage 3, T2DM, "
        "ICD in situ.",
        0, 1, 1, 1,
    ),

    # 142. Cortisol excess: thin skin, poor healing, but a short laparoscopic
    #      case in a young woman.
    (
        "40F, ASA 3, for laparoscopic left adrenalectomy. Indication: "
        "cortisol-secreting adrenal adenoma with Cushing syndrome. PMH: "
        "secondary HTN, steroid-induced diabetes, osteopenia, proximal myopathy.",
        0, 0, 0, 0,
    ),

    # 143. Neurodegenerative disease with bulbar involvement. He is already
    #      aspirating; the tube does not fix his swallow.
    (
        "64M, ASA 4, for open gastrostomy tube placement. Indication: bulbar "
        "amyotrophic lateral sclerosis with dysphagia and 25 lb weight loss. "
        "PMH: ALS diagnosed 2024, FVC 48% predicted, nocturnal BiPAP.",
        0, 1, 0, 0,
    ),

    # 144. DISCORDANT vs rows 1 and 28: hip fracture, but she is 68, fit, and
    #      the operation happens the same day.
    (
        "68F, ASA 2, for right hip intramedullary nailing. Indication: "
        "intertrochanteric fracture, cycling accident. PMH: hypothyroidism. "
        "Marathon runner, no comorbidities. To OR within 8h.",
        0, 0, 0, 0,
    ),

    # 145. Fulminant colitis with toxic physiology. Colectomy is a rescue and
    #      he is already in shock when he arrives.
    (
        "72M, ASA 5E, for emergent total abdominal colectomy with end "
        "ileostomy. Indication: fulminant Clostridioides difficile colitis with "
        "toxic megacolon, septic shock. PMH: recent prolonged antibiotics, CKD "
        "stage 3, T2DM. Lactate 5.8, on two pressors.",
        1, 1, 1, 1,
    ),

    # 146. Emphysema surgery in a patient who has almost no reserve to give
    #      away.
    (
        "63M, ASA 4, for bilateral lung volume reduction surgery via VATS. "
        "Indication: severe upper lobe predominant emphysema, FEV1 28% "
        "predicted, dyspnea at rest. PMH: COPD, former smoker 60 pack-years, "
        "home O2 continuous.",
        0, 1, 0, 0,
    ),

    # 147. Elderly but elective and low-impact; he walks in and walks out.
    (
        "77M, ASA 3, for open bilateral inguinal hernia repair with mesh. "
        "Indication: bilateral symptomatic inguinal hernias, enlarging. PMH: "
        "HTN, BPH, hyperlipidemia. Independent, plays golf. Regional planned.",
        0, 0, 0, 0,
    ),

    # 148. Difficult gallbladder converted to open. Longer case, subcostal
    #      incision, but a robust patient.
    (
        "57M, ASA 3, for laparoscopic cholecystectomy converted to open. "
        "Indication: gangrenous cholecystitis with dense Calot triangle "
        "inflammation. PMH: T2DM, BMI 36, HTN. Symptoms 6 days.",
        0, 0, 0, 0,
    ),

    # 149. Removing a failed graft in a patient back on dialysis. Already
    #      anuric; the AKI label has nowhere to go.
    (
        "48F, ASA 4, for transplant nephrectomy. Indication: chronic allograft "
        "failure with graft intolerance syndrome, fever and graft tenderness, "
        "back on hemodialysis. PMH: transplant 2014, ESRD, immunosuppression "
        "being withdrawn.",
        0, 0, 0, 0,
    ),

    # 150. Elective aesthetic surgery in a well patient.
    (
        "52F, ASA 1, for rhytidectomy with submental liposuction. Indication: "
        "elective facial rejuvenation. PMH: none. Nonsmoker, normotensive.",
        0, 0, 0, 0,
    ),

    # 151. Elective reduction in a healthy woman, day case.
    (
        "34F, ASA 2, for bilateral reduction mammoplasty. Indication: "
        "symptomatic macromastia with chronic back and shoulder pain, grooving. "
        "PMH: BMI 32. Nonsmoker.",
        0, 0, 0, 0,
    ),

    # 152. Small submandibular gland excision, healthy patient.
    (
        "49F, ASA 2, for left submandibular gland excision. Indication: "
        "recurrent sialadenitis with sialolithiasis, three prior admissions. "
        "PMH: Sjogren syndrome, hypothyroidism.",
        0, 0, 0, 0,
    ),

    # 153. Minor anorectal procedure in a young man.
    (
        "37M, ASA 1, for fistulotomy with seton placement. Indication: "
        "transsphincteric perianal fistula, recurrent. PMH: none. Outpatient.",
        0, 0, 0, 0,
    ),

    # 154. Accreta with catastrophic bleeding. Massive transfusion, prolonged
    #      hypotension, and a young kidney that still gives way.
    (
        "36F, ASA 4E, for cesarean hysterectomy. Indication: placenta percreta "
        "with bladder invasion, 4 L blood loss. PMH: three prior cesareans. "
        "Massive transfusion protocol, 12 units packed cells.",
        0, 0, 1, 0,
    ),

    # 155. Drainage of a tamponading effusion; short subxiphoid case, immediate
    #      hemodynamic improvement.
    (
        "58F, ASA 4E, for subxiphoid pericardial window. Indication: malignant "
        "pericardial effusion with tamponade physiology, pulsus paradoxus. PMH: "
        "metastatic breast cancer on palliative chemotherapy.",
        0, 0, 0, 0,
    ),

    # 156. Vascular tumor at the carotid bifurcation. Meticulous but clean; she
    #      is young and well.
    (
        "45F, ASA 2, for excision of left carotid body tumor, Shamblin II. "
        "Indication: enlarging paraganglioma with dysphagia. PMH: none. "
        "Normotensive, biochemically nonfunctional.",
        0, 0, 0, 0,
    ),

    # 157. Giant paraesophageal hernia in an elderly patient. The stomach has
    #      been in her chest for years and her lung has been splinted by it.
    (
        "80F, ASA 3, for laparoscopic paraesophageal hernia repair with Nissen "
        "fundoplication. Indication: giant type III hiatal hernia with "
        "postprandial dyspnea and anemia. PMH: HTN, mild COPD, osteoporosis.",
        0, 1, 0, 0,
    ),

    # 158. Complex pelvic ring injury. Long supine case, blood loss, and weeks
    #      of restricted weight-bearing ahead.
    (
        "43M, ASA 3E, for open reduction internal fixation of pelvic ring, "
        "anterior and posterior. Indication: unstable APC-III pelvic fracture "
        "after crush injury, s/p preperitoneal packing. PMH: none. 8 units "
        "transfused on day 1.",
        1, 0, 0, 0,
    ),

    # 159. Elective foot surgery, healthy adult, ankle block.
    (
        "44F, ASA 1, for right first metatarsophalangeal joint fusion. "
        "Indication: end-stage hallux rigidus with intractable pain. PMH: none. "
        "Ankle block planned.",
        0, 0, 0, 0,
    ),

    # 160. Young woman, urgent but brief laparoscopic case.
    (
        "26F, ASA 2E, for emergent laparoscopic detorsion with ovarian "
        "cystectomy. Indication: right ovarian torsion with 6 cm dermoid cyst, "
        "acute pain 8h. PMH: none. Ovary viable after detorsion.",
        0, 0, 0, 0,
    ),

    # 161. Elective hip in an obese but otherwise well patient.
    (
        "56M, ASA 3, for left total hip arthroplasty. Indication: end-stage "
        "osteoarthritis, failed conservative management. PMH: BMI 43, OSA on "
        "CPAP, HTN, T2DM diet-controlled.",
        1, 0, 0, 0,
    ),

    # 162. Solitary brain metastasis. Resection is short and she is
    #      cognitively intact going in.
    (
        "57F, ASA 3, for right occipital craniotomy for resection of solitary "
        "brain metastasis. Indication: 2.5 cm metastasis from NSCLC with "
        "vasogenic edema. PMH: stage IV NSCLC on osimertinib, on dexamethasone.",
        0, 0, 0, 0,
    ),

    # 163. Thyroid cancer with a formal neck dissection; longer case but a
    #      healthy young host.
    (
        "42F, ASA 2, for total thyroidectomy with central and left lateral neck "
        "dissection. Indication: papillary thyroid carcinoma with nodal "
        "metastases. PMH: none. Euthyroid, vocal cords normal preoperatively.",
        0, 0, 0, 0,
    ),

    # 164. Emergency surgical airway. He survives, but the aspiration during
    #      the obstruction seeds his lungs.
    (
        "59M, ASA 5E, for emergent cricothyroidotomy converted to tracheostomy. "
        "Indication: acute upper airway obstruction from supraglottic mass, "
        "cannot intubate cannot oxygenate. PMH: laryngeal cancer, current "
        "smoker, prior radiotherapy to neck.",
        0, 1, 0, 0,
    ),

    # 165. Small bowel tumour resection, laparoscopic, in a fit patient.
    (
        "58M, ASA 2, for laparoscopic small bowel resection. Indication: 5 cm "
        "jejunal gastrointestinal stromal tumour with occult bleeding. PMH: "
        "hyperlipidemia. Hemoglobin 10.9.",
        0, 0, 0, 0,
    ),

    # 166. Open bladder work in a man who retains; short and uncomplicated.
    (
        "66M, ASA 3, for open bladder diverticulectomy. Indication: large "
        "bladder diverticulum with recurrent infection and incomplete emptying. "
        "PMH: BPH, HTN, recurrent UTI. Prior TURP 2021.",
        0, 0, 0, 0,
    ),

    # 167. Two-level anterior cervical fusion, short, home the next day.
    (
        "51M, ASA 2, for C5-C7 anterior cervical discectomy and fusion. "
        "Indication: cervical spondylotic myelopathy with hand clumsiness and "
        "gait change. PMH: HTN, former smoker.",
        0, 0, 0, 0,
    ),

    # 168. Infected prosthesis with bacteremia in a diabetic. Washouts,
    #      spacers, and weeks of immobility.
    (
        "70F, ASA 4, for explant of infected right total knee arthroplasty with "
        "antibiotic spacer insertion. Indication: chronic periprosthetic joint "
        "infection, MRSA, sinus tract. PMH: T2DM, RA on tocilizumab, CKD stage "
        "3, BMI 37.",
        1, 0, 1, 0,
    ),

    # 169. Bypass surgery in a dialysis patient. Long pump run, diffuse
    #      disease, but the kidneys are already lost so AKI is off the table.
    (
        "65M, ASA 4, for coronary artery bypass grafting x4. Indication: left "
        "main and three-vessel disease, unstable angina. PMH: ESRD on "
        "hemodialysis, T2DM, PVD, prior CVA, current smoker.",
        0, 1, 0, 1,
    ),

    # 170. Young woman, short laparoscopic case, home same day.
    (
        "30F, ASA 1, for laparoscopic ovarian cystectomy. Indication: "
        "persistent 6 cm endometrioma with dysmenorrhea. PMH: endometriosis. "
        "Fertility-sparing intent.",
        0, 0, 0, 0,
    ),

    # 171. Minor scrotal case under local in a healthy man.
    (
        "48M, ASA 2, for left hydrocelectomy. Indication: symptomatic hydrocele "
        "with scrotal heaviness, recurrent after aspiration. PMH: HTN. "
        "Outpatient, local with sedation.",
        0, 0, 0, 0,
    ),

    # 172. Penetrating chest trauma in a young man. Thoracotomy, chest tubes,
    #      retained blood in a contused lung.
    (
        "22M, ASA 4E, for emergent left thoracotomy. Indication: stab wound to "
        "left chest with massive hemothorax, 1600 mL initial chest tube output. "
        "No PMH. Lung laceration repaired, 4 units transfused.",
        0, 1, 0, 0,
    ),

    # 173. Blunt liver injury in a middle-aged patient; packed, resuscitated,
    #      and closed at second look.
    (
        "47M, ASA 4E, for emergent laparotomy with hepatic packing. Indication: "
        "grade IV liver laceration after motor vehicle collision, hemodynamic "
        "instability. PMH: HTN. 8 units transfused, temporary closure.",
        0, 0, 1, 0,
    ),

    # 174. Revision bariatric surgery: adhesions, longer case, but a mobile
    #      patient who ambulates early.
    (
        "45F, ASA 3, for revision of sleeve gastrectomy to Roux-en-Y gastric "
        "bypass. Indication: intractable GERD with esophagitis after sleeve, "
        "weight regain. PMH: prior sleeve 2019, BMI 41, OSA, HTN.",
        0, 0, 0, 0,
    ),

    # 175. DISCORDANT: elderly and a fracture, but a wrist under regional and
    #      she goes home the same day, in her own bed by evening.
    (
        "83F, ASA 3, for open reduction internal fixation of left distal "
        "radius. Indication: displaced intra-articular distal radius fracture "
        "after fall. PMH: osteoporosis, HTN, hypothyroidism. Independent, "
        "regional block, day case.",
        0, 0, 0, 0,
    ),

    # 176. Large stone burden, prone, irrigation, and a patient who arrives
    #      with an infected system.
    (
        "54F, ASA 3, for right percutaneous nephrolithotomy. Indication: 3 cm "
        "staghorn calculus with recurrent infections. PMH: recurrent UTI, "
        "obesity BMI 36, T2DM. Preoperative urine culture positive, treated.",
        0, 0, 1, 0,
    ),

    # 177. Obstructing rectal tumour in an elderly patient; defunctioning
    #      stoma only, short case.
    (
        "78M, ASA 3E, for laparoscopic loop colostomy. Indication: obstructing "
        "low rectal adenocarcinoma, defunctioning prior to chemoradiation. PMH: "
        "HTN, CAD, mild cognitive impairment.",
        1, 0, 0, 1,
    ),

    # 178. Child with an anterior mediastinal mass; airway managed carefully,
    #      uneventful.
    (
        "12F, ASA 3, for anterior mediastinal mass biopsy via Chamberlain "
        "procedure. Indication: large anterior mediastinal mass, suspected "
        "lymphoma, no airway compression on CT. PMH: none. Spontaneous "
        "ventilation maintained.",
        0, 0, 0, 0,
    ),

    # 179. DISCORDANT vs rows 3 and 52: severe COPD, but the operation is a
    #      short peripheral one under regional and his chest is left alone.
    (
        "74M, ASA 4, for right below-knee amputation. Indication: chronic "
        "nonhealing heel ulcer with osteomyelitis, unreconstructable disease. "
        "PMH: severe COPD FEV1 40%, home O2, current smoker, PVD, T2DM. Spinal "
        "anesthesia.",
        1, 0, 0, 0,
    ),

    # 180. Young, well, brief laparoscopic case.
    (
        "27F, ASA 1, for laparoscopic cholecystectomy. Indication: symptomatic "
        "cholelithiasis with recurrent biliary colic. No PMH. Outpatient.",
        0, 0, 0, 0,
    ),

    # 181. DISCORDANT vs row 26: same cancer, same operation, but never-smoker,
    #      minimally invasive, and epidural analgesia. The chest holds.
    (
        "54M, ASA 2, for minimally invasive Ivor Lewis esophagectomy. "
        "Indication: distal esophageal adenocarcinoma, cT2N0. PMH: Barrett "
        "esophagus, never smoker, BMI 26. Thoracic epidural planned.",
        0, 0, 0, 0,
    ),

    # 182. Combined revascularisation and valve work: long pump run in a
    #      diabetic with stiff kidneys.
    (
        "72M, ASA 4, for coronary artery bypass grafting x2 with mitral valve "
        "replacement. Indication: ischemic mitral regurgitation with "
        "two-vessel disease, EF 35%. PMH: prior MI, T2DM on insulin, CKD stage "
        "3, HTN.",
        0, 0, 1, 1,
    ),

    # 183. Reconstruction after an iatrogenic bile duct injury; young patient,
    #      technically demanding but physiologically well tolerated.
    (
        "42F, ASA 3, for Roux-en-Y hepaticojejunostomy. Indication: Strasberg "
        "E2 bile duct injury after laparoscopic cholecystectomy 6 weeks ago, "
        "biliary stricture. PMH: recent cholecystectomy, percutaneous drain in "
        "situ.",
        0, 0, 0, 0,
    ),

    # 184. Splenectomy for a haematologic indication in a young woman on
    #      steroids; laparoscopic and straightforward.
    (
        "33F, ASA 2, for laparoscopic splenectomy. Indication: chronic immune "
        "thrombocytopenia refractory to steroids and rituximab, platelets 18. "
        "PMH: ITP 4 years, on prednisone. Vaccinated preoperatively.",
        0, 0, 0, 0,
    ),

    # 185. Cholecystitis in a demented nursing-home resident. Short case, but
    #      the admission itself is what undoes her.
    (
        "89F, ASA 4E, for laparoscopic cholecystectomy. Indication: acute "
        "gangrenous cholecystitis, septic. PMH: advanced dementia, AF on "
        "apixaban, CKD stage 3, prior CVA. Nursing home, needs assistance with "
        "all ADLs.",
        0, 0, 1, 1,
    ),

    # 186. Elective laparoscopic resection for cancer in a fit patient. Short,
    #      enhanced recovery, out on day 3. The malignancy still counts.
    (
        "62F, ASA 2, for laparoscopic left hemicolectomy. Indication: descending "
        "colon adenocarcinoma, T3N0 on staging. PMH: hypothyroidism, BMI 25. "
        "Enhanced recovery pathway.",
        1, 0, 0, 0,
    ),

    # 187. Elective foregut repair in a middle-aged patient.
    (
        "53F, ASA 2, for laparoscopic hiatal hernia repair with Toupet "
        "fundoplication. Indication: symptomatic type I hiatal hernia with "
        "refractory reflux. PMH: GERD, BMI 29, asthma.",
        0, 0, 0, 0,
    ),

    # 188. Arteriovenous malformation resection: young patient, eloquent
    #      cortex, but no baseline cognitive deficit and a clean course.
    (
        "31M, ASA 2, for right frontal craniotomy for AVM resection. "
        "Indication: Spetzler-Martin grade II AVM after single seizure, "
        "embolised preoperatively. PMH: seizure disorder on levetiracetam.",
        0, 0, 0, 0,
    ),

    # 189. Revision knee: long case, extensive dissection, prolonged tourniquet
    #      and slow mobilisation.
    (
        "69M, ASA 3, for revision left total knee arthroplasty. Indication: "
        "aseptic tibial component loosening with instability. PMH: BMI 38, "
        "T2DM, HTN, prior DVT after index arthroplasty 2018.",
        1, 0, 0, 0,
    ),

    # 190. Radical vulvar surgery with groin dissection in an elderly patient.
    (
        "77F, ASA 3, for radical vulvectomy with bilateral inguinofemoral "
        "lymphadenectomy. Indication: stage II vulvar squamous cell carcinoma. "
        "PMH: lichen sclerosus, HTN, T2DM, BMI 34. Prolonged immobility "
        "expected.",
        1, 0, 0, 0,
    ),

    # 191. Elective prosthetic implantation in a diabetic; short, clean.
    (
        "61M, ASA 3, for inflatable penile prosthesis implantation. Indication: "
        "medication-refractory erectile dysfunction after radical "
        "prostatectomy. PMH: T2DM well controlled, prostatectomy 2022, HTN.",
        0, 0, 0, 0,
    ),

    # 192. Young man, short inguinal case under general anesthesia.
    (
        "28M, ASA 1, for right radical inguinal orchiectomy. Indication: right "
        "testicular mass with elevated AFP and beta-hCG, suspected nonseminoma. "
        "No PMH. Sperm banking completed.",
        0, 0, 0, 0,
    ),

    # 193. Airway stenting in a patient whose tumour is already obstructing his
    #      bronchus; distal collapse is infected before he reaches the OR.
    (
        "68M, ASA 4, for rigid bronchoscopy with silicone airway stent "
        "placement. Indication: malignant central airway obstruction from "
        "recurrent NSCLC, 80% left main stenosis with post-obstructive "
        "collapse. PMH: NSCLC, COPD, current smoker.",
        0, 1, 0, 0,
    ),

    # 194. Complex valve surgery in a young athlete; long pump run but a
    #      pristine host.
    (
        "26M, ASA 3, for Ross procedure. Indication: severe aortic "
        "regurgitation from bicuspid valve with LV dilation, wishes to avoid "
        "anticoagulation. PMH: bicuspid aortic valve. Otherwise healthy, "
        "competitive cyclist.",
        0, 0, 0, 0,
    ),

    # 195. Big abdominal wall reconstruction with a free flap in an obese
    #      diabetic. Long case, immobile on the table and afterwards.
    (
        "58M, ASA 3, for abdominal wall reconstruction with free anterolateral "
        "thigh flap. Indication: complex ventral hernia with loss of domain "
        "after mesh explantation for infection. PMH: BMI 44, T2DM, four prior "
        "laparotomies.",
        1, 0, 0, 0,
    ),

    # 196. Ruptured ectopic in a young woman. Bleeding, but she is 27 and
    #      recovers immediately.
    (
        "27F, ASA 3E, for emergent laparoscopic salpingectomy. Indication: "
        "ruptured right tubal ectopic pregnancy with hemoperitoneum, "
        "tachycardic, hemoglobin 7.8. PMH: prior chlamydial PID. Two units "
        "transfused.",
        0, 0, 0, 0,
    ),

    # 197. Elective resection for benign stricture in a fit patient.
    (
        "59M, ASA 2, for laparoscopic sigmoid colectomy. Indication: "
        "symptomatic diverticular stricture after three episodes of "
        "diverticulitis. PMH: diverticular disease, HTN, BMI 30.",
        0, 0, 0, 0,
    ),

    # 198. Cystectomy with a neobladder: long, complex, urinary reconstruction
    #      with metabolic consequences.
    (
        "64M, ASA 3, for radical cystoprostatectomy with orthotopic ileal "
        "neobladder. Indication: muscle-invasive bladder cancer after "
        "neoadjuvant chemotherapy. PMH: HTN, former smoker 30 pack-years, CKD "
        "stage 2. Anticipated 7h case.",
        1, 0, 1, 0,
    ),

    # 199. Malignant middle cerebral artery infarction. He survives the
    #      decompression, but the stroke and the ICU stay define his course.
    (
        "56M, ASA 4E, for emergent decompressive hemicraniectomy. Indication: "
        "malignant right MCA infarction with midline shift 9 mm, declining GCS. "
        "PMH: AF not anticoagulated, HTN, current smoker.",
        1, 1, 0, 1,
    ),

    # 200. Elective forefoot surgery, healthy adult, ankle block, day case.
    (
        "39F, ASA 1, for right scarf osteotomy for hallux valgus. Indication: "
        "symptomatic bunion with difficulty in footwear. PMH: none. Ankle block "
        "planned.",
        0, 0, 0, 0,
    ),

    # 201. Duct exploration in an elderly patient with cholangitis; she arrives
    #      septic with a stone impacted at the ampulla.
    (
        "76F, ASA 4E, for open common bile duct exploration with T-tube. "
        "Indication: severe cholangitis with impacted stone, failed ERCP, "
        "hypotensive. PMH: HTN, CKD stage 3, AF. Bilirubin 9.4, WBC 19.",
        0, 0, 1, 1,
    ),

    # 202. Very old, very frail, hip fracture. The archetype at its most
    #      extreme.
    (
        "93F, ASA 4, for right hip hemiarthroplasty. Indication: displaced "
        "femoral neck fracture after fall from bed. PMH: advanced dementia, "
        "CHF, CKD stage 4, prior hip fracture 2023. Nursing home, hoist "
        "transfers.",
        0, 1, 1, 1,
    ),

    # 203. DISCORDANT: current smoker, but a knee under regional in an
    #      otherwise fit patient. Smoking alone does not give you pneumonia.
    (
        "62M, ASA 3, for right total knee arthroplasty. Indication: end-stage "
        "osteoarthritis. PMH: current smoker 25 pack-years, HTN, mild COPD not "
        "on inhalers. Spinal with adductor canal block, mobilised day 0.",
        0, 0, 0, 0,
    ),

    # 204. Healthy altruistic donor. Losing a kidney, but the remaining one is
    #      normal and he is 38.
    (
        "38M, ASA 1, for laparoscopic donor nephrectomy, left. Indication: "
        "living related kidney donation to sibling. No PMH. Creatinine 0.8, "
        "GFR 112, extensive workup normal.",
        0, 0, 0, 0,
    ),

    # 205. Walled-off necrosis debridement. He has been inflamed for weeks and
    #      is septic when he comes to theatre.
    (
        "49M, ASA 4, for open pancreatic necrosectomy with drainage. "
        "Indication: infected walled-off pancreatic necrosis 6 weeks after "
        "severe gallstone pancreatitis, failed step-up drainage. PMH: alcohol "
        "use, T2DM new-onset. Febrile, on antibiotics.",
        1, 1, 1, 0,
    ),

    # 206. Deep sternal wound infection in a diabetic. Reopening a healing
    #      sternum, muscle flap coverage, long recovery.
    (
        "67M, ASA 4, for sternal debridement with bilateral pectoralis major "
        "advancement flaps. Indication: deep sternal wound infection 3 weeks "
        "after CABG, MSSA. PMH: T2DM on insulin HbA1c 9.4, BMI 39, CKD stage 3, "
        "recent CABG.",
        0, 1, 0, 0,
    ),

    # 207. Chest wall sarcoma resection with reconstruction; the chest wall
    #      mechanics are disrupted by design.
    (
        "52M, ASA 3, for chest wall resection of ribs 4-7 with mesh and "
        "methylmethacrylate reconstruction. Indication: chondrosarcoma of the "
        "right chest wall. PMH: former smoker, HTN. Four ribs resected.",
        0, 1, 0, 0,
    ),

    # 208. Large soft tissue sarcoma in the thigh. Long resection, malignancy,
    #      and immobility afterwards.
    (
        "60F, ASA 3, for radical resection of left thigh soft tissue sarcoma "
        "with vastus lateralis excision. Indication: 14 cm high-grade "
        "undifferentiated pleomorphic sarcoma, s/p neoadjuvant radiotherapy. "
        "PMH: HTN, BMI 33.",
        1, 0, 0, 0,
    ),

    # 209. Free flap reconstruction after tumour resection in a smoker with a
    #      radiated neck; long case, tracheostomy, impaired swallow.
    (
        "63M, ASA 4, for composite resection of oral cavity tumour with free "
        "fibula flap reconstruction and tracheostomy. Indication: recurrent "
        "floor of mouth squamous cell carcinoma after prior radiotherapy. PMH: "
        "current smoker, alcohol use disorder, malnutrition.",
        1, 1, 0, 1,
    ),

    # 210. Adult tonsillectomy, healthy patient, day case.
    (
        "29F, ASA 1, for tonsillectomy. Indication: recurrent tonsillitis, six "
        "episodes this year, and peritonsillar abscess drained twice. No PMH.",
        0, 0, 0, 0,
    ),

    # 211. Elective bilateral oophorectomy in a well older woman.
    (
        "68F, ASA 2, for laparoscopic bilateral salpingo-oophorectomy. "
        "Indication: complex adnexal cyst on surveillance imaging, rising "
        "CA-125. PMH: HTN, prior hysterectomy for fibroids 2009.",
        0, 0, 0, 0,
    ),

    # 212. Anastomotic leak with peritonitis. He was recovering; now he is
    #      septic and back on the table.
    (
        "66M, ASA 4E, for emergent laparotomy with takedown of anastomosis and "
        "end colostomy. Indication: anastomotic leak with feculent peritonitis, "
        "day 6 after low anterior resection. PMH: rectal cancer, T2DM, HTN. On "
        "norepinephrine.",
        0, 1, 1, 1,
    ),

    # 213. Prophylactic colectomy in a young Lynch syndrome patient.
    (
        "34F, ASA 2, for laparoscopic total colectomy with ileorectal "
        "anastomosis. Indication: Lynch syndrome with synchronous colonic "
        "neoplasia, MLH1 pathogenic variant. PMH: none. Strong family history.",
        0, 0, 0, 0,
    ),

    # 214. Short device implantation under sedation in an elderly patient with
    #      heart failure.
    (
        "79M, ASA 4, for cardiac resynchronisation therapy defibrillator "
        "implantation. Indication: NYHA III heart failure, EF 28%, LBBB with "
        "QRS 168 ms. PMH: ischemic cardiomyopathy, CKD stage 3, T2DM. Sedation "
        "only.",
        0, 0, 0, 0,
    ),

    # 215. Aortobifemoral bypass: clamping the aorta above the renals for part
    #      of the case in a patient whose kidneys are already marginal.
    (
        "64M, ASA 4, for aortobifemoral bypass. Indication: aortoiliac occlusive "
        "disease with bilateral claudication at 50 m and tissue loss. PMH: "
        "current smoker 50 pack-years, CKD stage 3, COPD, CAD, HTN.",
        0, 1, 1, 0,
    ),

    # 216. Elective lap chole in an obese but well patient.
    (
        "46F, ASA 3, for laparoscopic cholecystectomy. Indication: symptomatic "
        "gallstones with recurrent biliary colic. PMH: BMI 45, T2DM, OSA on "
        "CPAP, HTN. Uneventful, home the same day.",
        0, 0, 0, 0,
    ),

    # 217. Rare hernia in a very thin elderly woman, presenting late with dead
    #      bowel.
    (
        "88F, ASA 4E, for emergent laparotomy with obturator hernia repair and "
        "small bowel resection. Indication: strangulated obturator hernia with "
        "ischemic ileum, 3 days of vomiting. PMH: BMI 17, dementia, HTN, "
        "chronic kidney disease.",
        0, 1, 1, 1,
    ),

    # 218. Femoral shaft fracture in a healthy young adult; nailed the same
    #      night, mobilising the next day.
    (
        "31M, ASA 2E, for right femoral shaft intramedullary nailing. "
        "Indication: closed midshaft femur fracture after motorcycle "
        "collision. PMH: none. Isolated injury, to OR within 12h.",
        0, 0, 0, 0,
    ),

    # 219. Hydrocephalus from a bleed. He is already obtunded; the drain is a
    #      holding measure and his head does not clear quickly.
    (
        "72M, ASA 4E, for external ventricular drain placement. Indication: "
        "obstructive hydrocephalus from intraventricular extension of "
        "hypertensive basal ganglia hemorrhage, GCS 9. PMH: uncontrolled HTN, "
        "T2DM, prior lacunar infarcts.",
        1, 1, 0, 1,
    ),

    # 220. Fertility-sparing surgery in a young woman; bloody but well
    #      tolerated.
    (
        "36F, ASA 2, for open abdominal myomectomy. Indication: multiple "
        "symptomatic fibroids with menorrhagia, desires future fertility. PMH: "
        "iron deficiency anemia, hemoglobin 9.8. Eight fibroids enucleated.",
        0, 0, 0, 0,
    ),

    # 221. DISCORDANT: 91 and having a joint replaced, but elective, regional,
    #      and she is cognitively intact and highly motivated.
    (
        "91F, ASA 3, for right total knee arthroplasty. Indication: end-stage "
        "osteoarthritis with intractable pain, still gardening. PMH: HTN, "
        "osteoporosis. Lives independently, MMSE 30. Spinal, mobilised day 0.",
        0, 0, 0, 0,
    ),

    # 222. Groin hernia obstructing in an otherwise well man; reduced viable
    #      bowel, home in two days.
    (
        "58M, ASA 2E, for emergent open right inguinal hernia repair. "
        "Indication: incarcerated inguinal hernia with obstruction, 10h of "
        "symptoms. PMH: none. Bowel viable after reduction, no resection.",
        0, 0, 0, 0,
    ),

    # 223. DISCORDANT vs row 34: same valve procedure, but not frail, normal
    #      kidneys, minimal contrast. Home on day 2.
    (
        "78M, ASA 3, for transcatheter aortic valve replacement, transfemoral. "
        "Indication: severe symptomatic aortic stenosis, intermediate risk. "
        "PMH: HTN, hyperlipidemia. Creatinine 0.9, gait speed normal, "
        "independent. Contrast 60 mL.",
        0, 0, 0, 0,
    ),

    # 224. DISCORDANT vs rows 8 and 42: same operation, but young, fit,
    #      minimally invasive, and no cancer cachexia.
    (
        "51F, ASA 2, for laparoscopic pancreaticoduodenectomy. Indication: "
        "neuroendocrine tumour of the pancreatic head, 2 cm, nonfunctioning. "
        "PMH: none. Albumin 4.1, no weight loss, never smoker.",
        0, 0, 0, 0,
    ),

    # 225. Young tall man with a primary pneumothorax; brief VATS, chest tube
    #      out on day 2.
    (
        "22M, ASA 1, for VATS bullectomy with mechanical pleurodesis. "
        "Indication: recurrent primary spontaneous right pneumothorax, third "
        "episode. PMH: none. Never smoker, BMI 19.",
        0, 0, 0, 0,
    ),

    # 226. Transsphenoidal resection: short, extracranial route, no
    #      craniotomy, and a well patient.
    (
        "45F, ASA 2, for endoscopic transsphenoidal resection of pituitary "
        "adenoma. Indication: nonfunctioning macroadenoma with bitemporal "
        "hemianopia. PMH: secondary hypothyroidism on replacement.",
        0, 0, 0, 0,
    ),

    # 227. Continence device in a man with a normal upper tract; short perineal
    #      case.
    (
        "70M, ASA 3, for artificial urinary sphincter implantation. Indication: "
        "severe stress incontinence after radical prostatectomy and salvage "
        "radiotherapy. PMH: prostate cancer, HTN, hyperlipidemia.",
        0, 0, 0, 0,
    ),

    # 228. Vaginal approach in a well older woman; no abdominal incision, fast
    #      recovery.
    (
        "63F, ASA 2, for vaginal hysterectomy with anterior and posterior "
        "repair. Indication: stage II uterovaginal prolapse with urinary "
        "symptoms. PMH: HTN, four vaginal deliveries. Regional planned.",
        0, 0, 0, 0,
    ),

    # 229. Appendicitis in the third trimester; laparoscopic, uneventful, fetus
    #      well.
    (
        "30F, ASA 2E, for laparoscopic appendectomy at 28 weeks gestation. "
        "Indication: acute appendicitis confirmed on MRI. PMH: uncomplicated "
        "pregnancy. Tocolysis not required, fetal monitoring reassuring.",
        0, 0, 0, 0,
    ),

    # 230. Young athlete, arthroscopic hip, day case.
    (
        "26M, ASA 1, for right hip arthroscopy with femoroacetabular "
        "impingement correction and labral repair. Indication: cam morphology "
        "with mechanical symptoms. PMH: none. Recreational footballer.",
        0, 0, 0, 0,
    ),

    # 231. Superficial venous surgery in a healthy patient; day case under
    #      tumescent local.
    (
        "47F, ASA 2, for endovenous laser ablation of great saphenous vein with "
        "phlebectomies. Indication: symptomatic varicose veins with lipodermato"
        "sclerosis, CEAP C4. PMH: two prior pregnancies, BMI 30.",
        0, 0, 0, 0,
    ),

    # 232. Thyroidectomy for Graves in a young woman; rendered euthyroid
    #      preoperatively.
    (
        "35F, ASA 2, for total thyroidectomy. Indication: Graves disease with "
        "large goitre, failed antithyroid drugs, mild orbitopathy. PMH: Graves "
        "disease. Euthyroid on carbimazole, Lugol iodine given.",
        0, 0, 0, 0,
    ),

    # 233. Coverage of a chronic wound in a diabetic; short case under
    #      regional, but the leg has been immobile for months.
    (
        "66M, ASA 4, for split-thickness skin grafting to left lower leg. "
        "Indication: chronic venous ulcer, 12 months duration, granulating "
        "after debridement. PMH: T2DM, PVD, CKD stage 3, obesity, prior DVT.",
        1, 0, 0, 0,
    ),

    # 234. Stripping the pericardium off a failing heart: bloody, slow, and the
    #      right heart struggles afterwards.
    (
        "61M, ASA 4, for radical pericardiectomy. Indication: constrictive "
        "pericarditis after prior mediastinal radiotherapy, refractory ascites "
        "and edema. PMH: Hodgkin lymphoma s/p mantle radiotherapy 1998, CKD "
        "stage 3, hepatic congestion.",
        0, 1, 1, 0,
    ),

    # 235. Dual organ transplant. Enteric drainage, heavy immunosuppression,
    #      and a long ICU stay.
    (
        "43M, ASA 4, for simultaneous pancreas-kidney transplant. Indication: "
        "type 1 diabetes with ESRD on peritoneal dialysis, hypoglycemia "
        "unawareness. PMH: T1DM 28 years, ESRD, gastroparesis, autonomic "
        "neuropathy.",
        1, 1, 1, 0,
    ),

    # 236. Crush injury with an unsalvageable limb. Rhabdomyolysis and
    #      pigment nephropathy in a young man.
    (
        "34M, ASA 4E, for emergent left above-knee amputation. Indication: "
        "mangled lower extremity after industrial crush injury, "
        "unreconstructable. No PMH. CK 42,000, myoglobinuria, 6 units "
        "transfused.",
        0, 0, 1, 0,
    ),

    # 237. Fistula takedown in a malnourished patient after multiple prior
    #      operations. Long adhesiolysis, but he pulls through.
    (
        "54M, ASA 3, for laparotomy with enterocutaneous fistula takedown and "
        "small bowel resection. Indication: high-output enterocutaneous fistula "
        "9 months after emergency laparotomy, on TPN. PMH: albumin 2.6, three "
        "prior laparotomies.",
        1, 0, 0, 0,
    ),

    # 238. Open enucleation for a very large prostate; more blood loss than a
    #      TURP but a well patient.
    (
        "72M, ASA 3, for open simple prostatectomy. Indication: 180 g prostate "
        "with refractory retention, unsuitable for TURP. PMH: HTN, "
        "hyperlipidemia, BPH. Independent, no cognitive complaints.",
        0, 0, 0, 0,
    ),

    # 239. Intradural tumour resection: long prone case, but a fit patient with
    #      no cognitive baseline issues.
    (
        "48F, ASA 2, for T6-T8 laminectomy with resection of intradural "
        "meningioma. Indication: progressive myelopathy with lower limb "
        "weakness. PMH: none. Ambulant with stick preoperatively.",
        0, 0, 0, 0,
    ),

    # 240. Child with appendicitis; laparoscopic, home the next day.
    (
        "11M, ASA 1, for laparoscopic appendectomy. Indication: acute "
        "appendicitis, 18h of periumbilical pain migrating to right iliac "
        "fossa. No PMH.",
        0, 0, 0, 0,
    ),

    # 241. Ex-premature infant with a hernia; caudal block, apnea monitoring
    #      overnight.
    (
        "4-month-old male, ASA 2, for right inguinal hernia repair. Indication: "
        "reducible inguinal hernia in ex-premature infant, born at 32 weeks. "
        "PMH: prematurity, resolved bronchopulmonary dysplasia. Caudal block.",
        0, 0, 0, 0,
    ),

    # 242. DISCORDANT: 92 years old, but topical anesthesia and twenty minutes.
    #      Age is not a physiologic insult by itself.
    (
        "92M, ASA 3, for left cataract extraction with intraocular lens. "
        "Indication: dense cataract with visual acuity 20/200, falls risk. PMH: "
        "AF on apixaban, HTN, mild hearing loss. Topical, sedation-free.",
        0, 0, 0, 0,
    ),

    # 243. Bypass surgery in a man with baseline cognitive impairment. Pump run
    #      plus dementia is a hard combination.
    (
        "81M, ASA 4, for coronary artery bypass grafting x3. Indication: left "
        "main disease with unstable angina, not suitable for PCI. PMH: mild "
        "dementia, HTN, T2DM, CKD stage 3, prior TIA.",
        0, 0, 1, 1,
    ),

    # 244. Laparoscopic liver resection in a fit patient; parenchymal-sparing,
    #      minimal blood loss.
    (
        "56M, ASA 2, for laparoscopic left lateral sectionectomy. Indication: "
        "symptomatic 8 cm hepatic adenoma with growth on surveillance. PMH: "
        "none. Nonsmoker, no cirrhosis, normal liver function.",
        0, 0, 0, 0,
    ),

    # 245. Esophagectomy in an elderly patient. Highest-risk operation in the
    #      chest, done to a 76-year-old.
    (
        "76M, ASA 4, for transhiatal esophagectomy. Indication: distal "
        "esophageal adenocarcinoma after chemoradiation. PMH: COPD, former "
        "smoker 40 pack-years, CAD, 18 lb weight loss, albumin 3.0.",
        0, 1, 0, 1,
    ),

    # 246. Severe preeclampsia with HELLP. The renal injury is part of the
    #      disease, not the operation.
    (
        "33F, ASA 3E, for emergent cesarean section at 31 weeks. Indication: "
        "severe preeclampsia with HELLP syndrome, platelets 62, rising "
        "creatinine. PMH: chronic hypertension. On magnesium sulphate, general "
        "anesthesia.",
        0, 0, 1, 0,
    ),

    # 247. Long fusion in a demented elderly patient; hours prone, opioids, and
    #      a baseline she cannot get back to.
    (
        "82F, ASA 4, for L2-S1 posterior decompression and instrumented fusion. "
        "Indication: degenerative scoliosis with severe stenosis, unable to "
        "walk 20 m. PMH: mild dementia, osteoporosis, HTN, CKD stage 3. "
        "Anticipated 5h prone.",
        1, 0, 1, 1,
    ),

    # 248. Ruptured aneurysm treated endovascularly. Faster and less morbid
    #      than open, but she is still in shock with a dye load.
    (
        "79F, ASA 5E, for emergent endovascular repair of ruptured abdominal "
        "aortic aneurysm. Indication: ruptured 7 cm infrarenal AAA, "
        "hypotensive, aortic balloon occlusion used. PMH: HTN, COPD, CKD stage "
        "3.",
        0, 0, 1, 1,
    ),

    # 249. Deep neck space infection in a diabetic; drained, airway secured,
    #      recovers.
    (
        "51M, ASA 3E, for incision and drainage of deep neck abscess with "
        "tracheostomy. Indication: Ludwig angina from dental source with airway "
        "compromise. PMH: T2DM HbA1c 9.8, poor dentition, smoker.",
        0, 1, 0, 0,
    ),

    # 250. Straightforward stoma closure in a recovered patient.
    (
        "48M, ASA 2, for loop ileostomy closure. Indication: restoration of "
        "continuity 5 months after low anterior resection, contrast study "
        "normal. PMH: rectal cancer in remission, BMI 27.",
        0, 0, 0, 0,
    ),

    # 251. Repeat bladder resection in an older patient; short, spinal, day
    #      case.
    (
        "79M, ASA 3, for transurethral resection of bladder tumour. Indication: "
        "recurrent high-grade non-muscle-invasive bladder cancer on "
        "surveillance cystoscopy. PMH: HTN, CKD stage 3, former smoker. "
        "Independent.",
        0, 0, 0, 0,
    ),

    # 252. Redo sternotomy for a failing prosthesis in a patient with
    #      endocarditis; long pump run, embolic load, nephrotoxic antibiotics.
    (
        "59M, ASA 4, for redo aortic valve replacement. Indication: prosthetic "
        "valve endocarditis with root abscess, vegetation 14 mm. PMH: AVR 2019, "
        "IV drug use, hepatitis C, CKD stage 2. On gentamicin and vancomycin.",
        0, 0, 1, 1,
    ),

    # 253. Recurrent hernia after two prior repairs; longer dissection but a
    #      well patient.
    (
        "55M, ASA 2, for open recurrent right inguinal hernia repair with "
        "preperitoneal mesh. Indication: second recurrence after two prior "
        "repairs. PMH: none. Manual labourer, BMI 26.",
        0, 0, 0, 0,
    ),

    # 254. Posterior fossa decompression in a young woman; short, no cognitive
    #      baseline problems.
    (
        "29F, ASA 2, for suboccipital craniectomy with C1 laminectomy for Chiari "
        "I decompression. Indication: symptomatic Chiari malformation with "
        "cough headache and syrinx. PMH: none.",
        0, 0, 0, 0,
    ),

    # 255. DISCORDANT vs rows 1, 28, 80 and 202: hip fracture, but he is 70,
    #      fit, cognitively sharp, and operated the same day.
    (
        "70M, ASA 2, for left hip hemiarthroplasty. Indication: displaced "
        "femoral neck fracture after a fall while hiking. PMH: hyperlipidemia. "
        "Independent, works full-time, no cognitive complaints. To OR within "
        "10h.",
        0, 0, 0, 0,
    ),

    # 256. Feeding tube in advanced dementia; short case, but she is already
    #      aspirating and confused at baseline.
    (
        "86F, ASA 4, for percutaneous endoscopic gastrostomy. Indication: "
        "advanced dementia with dysphagia and recurrent aspiration, family "
        "requesting feeding tube. PMH: advanced dementia, prior aspiration "
        "pneumonia x2, contractures.",
        0, 1, 0, 1,
    ),

    # 257. Acute limb ischemia from an embolus. Reperfusion injury and myoglobin
    #      hit the kidneys.
    (
        "75F, ASA 4E, for emergent left femoral embolectomy with fasciotomy. "
        "Indication: acute limb ischemia, Rutherford IIb, 8h of symptoms. PMH: "
        "AF not anticoagulated, CHF, CKD stage 3. Reperfusion syndrome, CK "
        "12,000.",
        0, 0, 1, 0,
    ),

    # 258. Elective hysterectomy for benign disease in a healthy woman.
    (
        "45F, ASA 2, for total laparoscopic hysterectomy. Indication: "
        "adenomyosis with chronic pelvic pain and menorrhagia, failed hormonal "
        "management. PMH: iron deficiency anemia, BMI 28. Ovaries preserved.",
        0, 0, 0, 0,
    ),

    # 259. Pleurodesis for a malignant effusion in a patient with a burdened
    #      chest.
    (
        "64F, ASA 4, for VATS talc pleurodesis with indwelling pleural "
        "catheter. Indication: recurrent malignant pleural effusion from "
        "metastatic breast cancer, three prior drainages. PMH: metastatic "
        "breast cancer, prior chest radiotherapy, dyspnea at rest.",
        0, 1, 0, 0,
    ),

    # 260. Minor drainage in a healthy young adult under sedation.
    (
        "32M, ASA 1, for incision and drainage of perianal abscess. Indication: "
        "acute perianal abscess with fluctuance, no fistula identified. No PMH. "
        "Outpatient.",
        0, 0, 0, 0,
    ),

    # 261. Perforated appendicitis in a child; washed out, antibiotics, home in
    #      four days.
    (
        "9F, ASA 2E, for laparoscopic appendectomy with peritoneal lavage. "
        "Indication: perforated appendicitis with pelvic collection, 4 days of "
        "symptoms. PMH: none. Febrile, WBC 21.",
        0, 0, 0, 0,
    ),

    # 262. Myectomy in a fit patient with hypertrophic cardiomyopathy; short
    #      pump run, good ventricle.
    (
        "47M, ASA 3, for transaortic septal myectomy. Indication: obstructive "
        "hypertrophic cardiomyopathy with 90 mmHg gradient and exertional "
        "syncope, failed medical therapy. PMH: HOCM, otherwise well. EF 70%.",
        0, 0, 0, 0,
    ),

    # 263. Neuromodulation device in a well patient; short, prone, sedation.
    (
        "53F, ASA 2, for sacral neuromodulation device implantation. "
        "Indication: refractory urgency urinary incontinence, successful "
        "percutaneous trial. PMH: none relevant. Day case.",
        0, 0, 0, 0,
    ),

    # 264. Short cervical procedure in a healthy pregnancy.
    (
        "32F, ASA 2, for McDonald cervical cerclage at 14 weeks. Indication: "
        "history of two second-trimester losses with painless cervical "
        "dilation. PMH: cervical insufficiency. Spinal anesthesia.",
        0, 0, 0, 0,
    ),

    # 265. Palliative bypass in a cachectic patient with obstructing pancreatic
    #      cancer.
    (
        "69M, ASA 4, for open gastrojejunostomy. Indication: gastric outlet "
        "obstruction from unresectable pancreatic head adenocarcinoma, "
        "palliative. PMH: metastatic pancreatic cancer, 30 lb weight loss, "
        "albumin 2.4, on chemotherapy.",
        1, 0, 0, 0,
    ),

    # 266. Shoulder fracture in an elderly woman; regional, sling, discharged
    #      day 2 to her own home.
    (
        "81F, ASA 3, for right shoulder hemiarthroplasty. Indication: four-part "
        "proximal humerus fracture after fall. PMH: osteoporosis, HTN, "
        "hypothyroidism. Independent, lives with husband, cognitively intact.",
        0, 0, 0, 0,
    ),

    # 267. Awake deep brain stimulation in a Parkinson patient; long but awake,
    #      and cognitively screened before listing.
    (
        "64M, ASA 3, for bilateral subthalamic nucleus deep brain stimulator "
        "implantation. Indication: Parkinson disease with motor fluctuations "
        "and dyskinesia, levodopa-responsive. PMH: Parkinson disease 9 years. "
        "Neuropsychology screening normal.",
        0, 0, 0, 0,
    ),

    # 268. Renal revascularisation in a patient whose kidneys are the problem;
    #      clamping the renal artery in an already-injured kidney.
    (
        "62F, ASA 4, for aortorenal bypass, left. Indication: renal artery "
        "stenosis with flash pulmonary edema and declining function, failed "
        "stenting. PMH: CKD stage 4 Cr 2.8, refractory HTN on five agents, "
        "CAD.",
        0, 0, 1, 0,
    ),

    # 269. Middle ear surgery in a healthy adult; day case.
    (
        "38M, ASA 1, for right cortical mastoidectomy with tympanoplasty. "
        "Indication: chronic suppurative otitis media with cholesteatoma. No "
        "PMH.",
        0, 0, 0, 0,
    ),

    # 270. Hand injury in a young adult; regional block, day case.
    (
        "25M, ASA 1, for repair of flexor digitorum profundus laceration, zone "
        "II, right index finger. Indication: laceration from broken glass. No "
        "PMH. Supraclavicular block.",
        0, 0, 0, 0,
    ),

    # 271. Fistula between airway and pleura after a lobectomy. Contaminated
    #      chest, muscle flap, and an already-poor lung.
    (
        "66M, ASA 4, for thoracotomy with bronchopleural fistula repair and "
        "intercostal muscle flap. Indication: bronchopleural fistula with "
        "empyema 3 weeks after right lower lobectomy. PMH: COPD, former smoker, "
        "recent lobectomy, malnutrition.",
        1, 1, 0, 0,
    ),

    # 272. Hernia in a transplant recipient on tacrolimus; clean case but the
    #      graft function is precarious.
    (
        "57M, ASA 4, for open incisional hernia repair with mesh. Indication: "
        "symptomatic incisional hernia at prior transplant incision. PMH: "
        "kidney transplant 2019 on tacrolimus and mycophenolate, baseline Cr "
        "1.6, T2DM, HTN.",
        0, 0, 1, 0,
    ),

    # 273. Retransplantation for graft failure; hostile abdomen, massive
    #      transfusion, and an encephalopathic patient going in.
    (
        "48F, ASA 5, for orthotopic liver retransplant. Indication: hepatic "
        "artery thrombosis with graft failure, day 9 after primary transplant. "
        "PMH: primary sclerosing cholangitis, recent transplant, encephalopathy "
        "grade 2. 22 units transfused.",
        0, 1, 1, 1,
    ),

    # 274. Polytrauma in a young patient. External fixation, ICU, chest
    #      contusion.
    (
        "29M, ASA 4E, for external fixation of bilateral femoral fractures and "
        "pelvic binder application. Indication: polytrauma after fall from "
        "height, injury severity score 34, bilateral pulmonary contusions. No "
        "PMH. Intubated.",
        1, 1, 0, 0,
    ),

    # 275. Perforated gastric ulcer in a frail elderly woman; late
    #      presentation, contaminated abdomen.
    (
        "84F, ASA 4E, for emergent laparotomy with omental patch repair. "
        "Indication: perforated gastric ulcer with generalised peritonitis, 2 "
        "days of pain. PMH: chronic NSAID use, HTN, CKD stage 3, mild dementia. "
        "Hypotensive on arrival.",
        0, 1, 1, 1,
    ),

    # 276. Reconstructive urology in a young patient; laparoscopic, short.
    (
        "27F, ASA 1, for laparoscopic dismembered pyeloplasty. Indication: "
        "symptomatic left pelviureteric junction obstruction with recurrent "
        "loin pain, split function 38%. PMH: none.",
        0, 0, 0, 0,
    ),

    # 277. Brief diagnostic gynecologic procedure in an elderly patient.
    (
        "74F, ASA 3, for hysteroscopy with dilatation and curettage. "
        "Indication: postmenopausal bleeding with thickened endometrium on "
        "ultrasound. PMH: HTN, T2DM, BMI 36. Day case, LMA.",
        0, 0, 0, 0,
    ),

    # 278. Pathological fracture from metastatic disease. Cancer plus a nailed
    #      femur plus limited mobility.
    (
        "68F, ASA 4, for prophylactic intramedullary nailing of left femur. "
        "Indication: impending pathological fracture from lytic metastasis, "
        "Mirels score 10. PMH: metastatic breast cancer on palliative "
        "chemotherapy, bone metastases, prior radiotherapy.",
        1, 0, 0, 0,
    ),

    # 279. Bloodless cardiac surgery in a fit patient; meticulous but
    #      uncomplicated.
    (
        "55F, ASA 3, for aortic valve replacement with mechanical prosthesis, "
        "no blood products. Indication: severe aortic stenosis, bicuspid valve. "
        "PMH: Jehovah's Witness, declines transfusion. Preoperative "
        "erythropoietin and iron, hemoglobin 14.2.",
        0, 0, 0, 0,
    ),

    # 280. Extreme obesity with pulmonary hypertension; the chest cannot cope
    #      with an abdomen this size after surgery.
    (
        "44M, ASA 4, for laparoscopic sleeve gastrectomy. Indication: class III "
        "obesity BMI 68 with obesity hypoventilation syndrome. PMH: OSA on "
        "BiPAP, pulmonary hypertension, T2DM, immobile, CO2 retention on ABG.",
        1, 1, 0, 0,
    ),

    # 281. Neurologically impaired child with reflux and recurrent aspiration.
    (
        "7M, ASA 4, for laparoscopic Nissen fundoplication with gastrostomy. "
        "Indication: severe GERD with recurrent aspiration in cerebral palsy. "
        "PMH: spastic quadriplegic cerebral palsy, epilepsy, scoliosis, two "
        "prior aspiration pneumonias.",
        0, 1, 0, 0,
    ),

    # 282. Chronic subdural in an anticoagulated elderly man; burr holes under
    #      local, but he was already confused before he arrived.
    (
        "83M, ASA 3, for bilateral burr hole drainage of chronic subdural "
        "hematomas. Indication: bilateral chronic SDH with progressive "
        "confusion and gait decline over 6 weeks. PMH: AF on apixaban, mild "
        "cognitive impairment, HTN, prior falls.",
        1, 0, 0, 1,
    ),

    # 283. Thoracic endograft: contrast, spinal drain, and a stiff kidney.
    (
        "71M, ASA 4, for thoracic endovascular aortic repair. Indication: 6.4 "
        "cm descending thoracic aortic aneurysm. PMH: HTN, COPD, CKD stage 3 Cr "
        "1.8, former smoker 45 pack-years. Lumbar drain placed, contrast 120 "
        "mL.",
        0, 0, 1, 0,
    ),

    # 284. Reoperative neck exploration for persistent disease; short, focused.
    (
        "58F, ASA 3, for reoperative parathyroidectomy. Indication: persistent "
        "primary hyperparathyroidism after failed initial exploration, ectopic "
        "mediastinal adenoma on imaging. PMH: nephrolithiasis, osteoporosis, "
        "CKD stage 2.",
        0, 0, 0, 0,
    ),

    # 285. Retroperitoneal sarcoma with multivisceral resection: hours long,
    #      bloody, and a kidney comes out with it.
    (
        "61M, ASA 3, for resection of retroperitoneal liposarcoma with left "
        "nephrectomy and partial colectomy. Indication: 24 cm well-"
        "differentiated liposarcoma. PMH: HTN, BMI 31. Anticipated 6h, EBL "
        "2000 mL.",
        1, 0, 1, 0,
    ),

    # 286. Reconstructive bladder surgery in a young adult with spina bifida;
    #      long but well tolerated.
    (
        "24F, ASA 3, for ileocystoplasty with Mitrofanoff channel. Indication: "
        "neurogenic bladder with poor compliance and upper tract deterioration "
        "in spina bifida. PMH: myelomeningocele, wheelchair user, VP shunt, "
        "recurrent UTI.",
        1, 0, 0, 0,
    ),

    # 287. Elbow arthroplasty in a rheumatoid patient; short, regional.
    (
        "66F, ASA 3, for right total elbow arthroplasty. Indication: end-stage "
        "rheumatoid arthropathy of the elbow with fixed flexion. PMH: "
        "rheumatoid arthritis on methotrexate and etanercept, osteoporosis, "
        "hypothyroidism.",
        0, 0, 0, 0,
    ),

    # 288. Off-pump grafting in a patient with a hostile aorta; avoids the pump
    #      and the kidneys are spared.
    (
        "70M, ASA 4, for off-pump coronary artery bypass grafting x2. "
        "Indication: two-vessel disease with heavily calcified ascending aorta, "
        "avoiding cross-clamp. PMH: CKD stage 3, prior CVA, PVD, HTN, former "
        "smoker.",
        0, 0, 0, 0,
    ),

    # 289. Radical resection for gallbladder cancer with liver bed clearance;
    #      long case in a jaundiced patient.
    (
        "65F, ASA 3, for radical cholecystectomy with segment IVb/V liver "
        "resection and portal lymphadenectomy. Indication: incidental "
        "gallbladder adenocarcinoma T2 on cholecystectomy specimen. PMH: T2DM, "
        "HTN, BMI 33.",
        1, 0, 0, 0,
    ),

    # 290. Sleeve resection preserving lung tissue; a bronchial anastomosis in
    #      a smoker's airway.
    (
        "60M, ASA 3, for right upper lobe sleeve lobectomy. Indication: "
        "centrally located NSCLC involving the right upper lobe bronchus, "
        "preserving parenchyma. PMH: COPD FEV1 62%, current smoker 35 "
        "pack-years, HTN.",
        0, 1, 0, 0,
    ),

    # 291. Infected labour in an obese patient; delivered urgently, febrile,
    #      but she is 26 and recovers.
    (
        "26F, ASA 3E, for emergent cesarean section. Indication: "
        "chorioamnionitis with maternal fever 39.1 and fetal tachycardia, "
        "arrest of dilation. PMH: BMI 44, gestational diabetes on insulin. "
        "Spinal anesthesia.",
        0, 0, 0, 0,
    ),

    # 292. Cauda equina: emergent decompression in a working-age patient; back
    #      on his feet quickly.
    (
        "44M, ASA 2E, for emergent L4-L5 decompression and discectomy. "
        "Indication: cauda equina syndrome with urinary retention and saddle "
        "anesthesia, symptoms 14h. PMH: none. To OR within 4h of diagnosis.",
        0, 0, 0, 0,
    ),

    # 293. Carotid surgery after a stroke; the brain is already injured but the
    #      operation itself is short and clean.
    (
        "72M, ASA 3, for right carotid endarterectomy. Indication: symptomatic "
        "90% internal carotid stenosis after right hemispheric TIA 8 days ago. "
        "PMH: HTN, hyperlipidemia, T2DM, former smoker. Independent.",
        0, 0, 0, 0,
    ),

    # 294. Frozen abdomen from prior sepsis; six hours of sharp dissection with
    #      enterotomies.
    (
        "58F, ASA 3, for laparotomy with extensive adhesiolysis and small bowel "
        "resection. Indication: recurrent adhesive obstruction after "
        "hysterectomy and prior peritonitis, four prior admissions. PMH: three "
        "prior laparotomies, BMI 24. Two enterotomies repaired.",
        0, 0, 0, 0,
    ),

    # 295. Losing the only kidney he has. Immediate dialysis dependence is the
    #      expected outcome, not an accident.
    (
        "67M, ASA 4, for partial nephrectomy in a solitary right kidney. "
        "Indication: 5 cm renal cell carcinoma in solitary kidney after prior "
        "left nephrectomy. PMH: prior nephrectomy 2018, CKD stage 3b Cr 2.2, "
        "HTN, T2DM. Warm ischemia 24 min.",
        0, 0, 1, 0,
    ),

    # 296. Planned tracheostomy in a long-stay ICU patient with a neurological
    #      injury and an unprotected airway.
    (
        "58M, ASA 4, for open tracheostomy. Indication: prolonged ventilation "
        "after severe traumatic brain injury, day 11, failed extubation twice. "
        "PMH: TBI with diffuse axonal injury, aspiration on admission, GCS 9T.",
        1, 1, 0, 1,
    ),

    # 297. Hip fracture two weeks after a myocardial infarction. Demand
    #      ischemia, held antiplatelets, and a stunned ventricle.
    (
        "79M, ASA 4E, for right hip hemiarthroplasty. Indication: displaced "
        "femoral neck fracture, fall 16 days after NSTEMI and PCI. PMH: recent "
        "NSTEMI on dual antiplatelet therapy, EF 40%, CKD stage 3, T2DM.",
        0, 0, 1, 1,
    ),

    # 298. Diversion in a paraplegic patient with a pressure sore; already
    #      immobile, already wheelchair-bound.
    (
        "46M, ASA 4, for laparoscopic end colostomy formation. Indication: "
        "faecal diversion for stage IV sacral pressure ulcer with "
        "osteomyelitis. PMH: T4 paraplegia 12 years, neurogenic bowel, "
        "recurrent UTI, prior DVT.",
        1, 0, 0, 0,
    ),

    # 299. Valve surgery in a dialysis patient with calcified everything.
    #      Already anuric, so the kidneys cannot get worse.
    (
        "62M, ASA 4, for aortic valve replacement with bioprosthesis. "
        "Indication: severe calcific aortic stenosis with syncope. PMH: ESRD on "
        "hemodialysis 7 years, secondary hyperparathyroidism, T2DM, PVD, "
        "anemia.",
        0, 1, 0, 1,
    ),

    # 300. Trivial day case in a well patient.
    (
        "41M, ASA 1, for excision of subcutaneous lipoma, posterior neck. "
        "Indication: enlarging 6 cm lipoma, cosmetically bothersome and "
        "catching on collars. No PMH. Local anesthesia.",
        0, 0, 0, 0,
    ),

    # 301. Short arthroscopic case in a healthy adult; day case.
    (
        "42M, ASA 1, for right knee arthroscopy with partial medial "
        "meniscectomy. Indication: bucket-handle meniscal tear with locking. No "
        "PMH. Recreational squash player.",
        0, 0, 0, 0,
    ),

    # 302. Device exchange for pump thrombosis; redo sternotomy in a patient
    #      with haemolysis and a marginal right heart.
    (
        "57M, ASA 4, for left ventricular assist device pump exchange. "
        "Indication: device thrombosis with haemolysis, LDH 2400, unresponsive "
        "to thrombolysis. PMH: LVAD 2024, ischemic cardiomyopathy, CKD stage 3, "
        "on anticoagulation.",
        0, 0, 1, 1,
    ),

    # 303. DISCORDANT vs row 3: same operation, but never-smoked, normal PFTs,
    #      epidural, and a young lung.
    (
        "48F, ASA 2, for VATS right upper lobectomy. Indication: 2 cm "
        "adenocarcinoma, cT1bN0, never smoker with EGFR mutation. PMH: none. "
        "FEV1 98% predicted, thoracic epidural.",
        0, 0, 0, 0,
    ),

    # 304. Endoscopic stone clearance from the bladder; short, spinal.
    (
        "68M, ASA 3, for transurethral cystolitholapaxy. Indication: 3 cm "
        "bladder calculus with chronic retention and recurrent UTI. PMH: BPH, "
        "HTN, T2DM. Spinal, day case.",
        0, 0, 0, 0,
    ),

    # 305. Laparoscopic excision of endometriosis in a young woman; long but
    #      physiologically light.
    (
        "34F, ASA 2, for laparoscopic excision of deep infiltrating "
        "endometriosis with ureterolysis. Indication: stage IV endometriosis "
        "with dyschezia and dyspareunia. PMH: endometriosis, prior "
        "laparoscopy 2022.",
        0, 0, 0, 0,
    ),

    # 306. Isolated clavicle fracture in a healthy young adult; day case.
    (
        "28M, ASA 1, for open reduction internal fixation of right clavicle. "
        "Indication: displaced midshaft clavicle fracture with 2 cm shortening "
        "after cycling fall. No PMH.",
        0, 0, 0, 0,
    ),

    # 307. Nerve decompression under regional; twenty minutes.
    (
        "55M, ASA 2, for left ulnar nerve decompression and anterior "
        "transposition at the elbow. Indication: cubital tunnel syndrome with "
        "intrinsic wasting, confirmed on nerve conduction. PMH: T2DM, HTN.",
        0, 0, 0, 0,
    ),

    # 308. Extra-anatomic bypass in a man with a hostile abdomen and a bad
    #      chest; tunnelled subcutaneously to avoid the aorta.
    (
        "73M, ASA 4, for axillobifemoral bypass. Indication: aortoiliac "
        "occlusion with bilateral critical limb ischemia, hostile abdomen after "
        "prior aortic graft infection. PMH: COPD FEV1 45%, current smoker, CAD, "
        "CKD stage 3.",
        0, 1, 0, 0,
    ),

    # 309. Voice surgery in a well patient; short, awake portion.
    (
        "50F, ASA 2, for type I thyroplasty with medialisation implant. "
        "Indication: left vocal cord paralysis with breathy dysphonia after "
        "thyroidectomy. PMH: prior thyroidectomy 2024, hypothyroidism on "
        "replacement.",
        0, 0, 0, 0,
    ),

    # 310. Contracture release and grafting in a burns survivor; young, mobile.
    (
        "31F, ASA 2, for release of axillary burn contracture with "
        "full-thickness skin graft. Indication: restrictive scar contracture "
        "limiting shoulder abduction, 2 years after flame burn. PMH: prior 22% "
        "TBSA burn.",
        0, 0, 0, 0,
    ),

    # 311. Bowel resection for endometriosis in a young woman; long
    #      laparoscopic case, quick recovery.
    (
        "37F, ASA 2, for laparoscopic anterior resection for rectosigmoid "
        "endometriosis. Indication: obstructing rectosigmoid endometriotic "
        "nodule with cyclical bowel symptoms. PMH: endometriosis, two prior "
        "laparoscopies.",
        0, 0, 0, 0,
    ),

    # 312. DISCORDANT vs row 61: living donor kidney, short cold ischemia,
    #      immediate graft function. Passed urine on the table.
    (
        "39F, ASA 4, for living related donor kidney transplant. Indication: "
        "ESRD from IgA nephropathy, preemptive transplant before dialysis. PMH: "
        "IgA nephropathy, HTN. Cold ischemia 45 min, HLA 4/6 match.",
        0, 0, 0, 0,
    ),

    # 313. Splenectomy after blunt trauma in an older patient; robust enough to
    #      recover cleanly.
    (
        "64M, ASA 3E, for emergent splenectomy. Indication: grade IV splenic "
        "injury with ongoing bleeding after fall from ladder. PMH: HTN, "
        "hyperlipidemia. Three units transfused, no other injuries.",
        0, 0, 0, 0,
    ),

    # 314. Incidental Meckel's in a young patient; brief laparoscopic case.
    (
        "19M, ASA 1, for laparoscopic Meckel diverticulectomy. Indication: "
        "symptomatic Meckel diverticulum with painless rectal bleeding, "
        "positive scan. No PMH.",
        0, 0, 0, 0,
    ),

    # 315. Urethral reconstruction with a buccal graft; long lithotomy but a
    #      well patient.
    (
        "45M, ASA 2, for buccal mucosa graft urethroplasty. Indication: 4 cm "
        "bulbar urethral stricture, recurrent after two dilatations. PMH: prior "
        "urethral trauma. Otherwise well.",
        0, 0, 0, 0,
    ),

    # 316. Interval debulking after chemotherapy; advanced cancer, long
    #      abdominal case, deconditioned patient.
    (
        "68F, ASA 3, for interval cytoreductive surgery with total abdominal "
        "hysterectomy, bilateral salpingo-oophorectomy, omentectomy and "
        "diaphragmatic stripping. Indication: stage IIIC ovarian cancer after "
        "three cycles of carboplatin-paclitaxel. PMH: HTN, albumin 3.1.",
        1, 1, 0, 0,
    ),

    # 317. Humerus fracture in an elderly woman; plated, sling, home in two
    #      days. Cognitively intact.
    (
        "78F, ASA 3, for open reduction internal fixation of left distal "
        "humerus. Indication: comminuted intra-articular distal humerus "
        "fracture after fall on ice. PMH: osteoporosis, HTN, hypothyroidism. "
        "Lives alone, independent.",
        0, 0, 0, 0,
    ),

    # 318. Isolated tricuspid surgery in a congested patient; hepatic and renal
    #      congestion precede the operation.
    (
        "66F, ASA 4, for tricuspid valve replacement. Indication: severe "
        "tricuspid regurgitation with right heart failure, refractory ascites "
        "and peripheral edema. PMH: prior mitral valve replacement 2016, AF, "
        "cardiac cirrhosis, CKD stage 3.",
        0, 0, 1, 0,
    ),

    # 319. Shunt revision in a young adult; short case, quick recovery.
    (
        "23F, ASA 2, for ventriculoperitoneal shunt revision, distal catheter. "
        "Indication: shunt malfunction with headache and vomiting, distal "
        "catheter disconnection on shunt series. PMH: congenital hydrocephalus, "
        "shunt since infancy.",
        0, 0, 0, 0,
    ),

    # 320. Popliteal aneurysm with distal embolisation; bypass and thrombolysis
    #      in a smoker, but the leg and the man both do well.
    (
        "68M, ASA 3, for right popliteal aneurysm exclusion with femoral-distal "
        "bypass. Indication: 3.2 cm popliteal aneurysm with distal "
        "embolisation. PMH: HTN, former smoker, contralateral popliteal "
        "aneurysm repaired 2023.",
        0, 0, 0, 0,
    ),

    # 321. Large retrosternal goitre; the trachea has been compressed for
    #      years and softens after removal.
    (
        "72F, ASA 3, for total thyroidectomy with sternotomy for retrosternal "
        "extension. Indication: large multinodular goitre with tracheal "
        "deviation and 60% narrowing, stridor on exertion. PMH: HTN, AF, "
        "obesity BMI 34.",
        0, 1, 0, 0,
    ),

    # 322. Minor stoma revision in a well patient; local plus sedation.
    (
        "52F, ASA 2, for revision of end ileostomy. Indication: parastomal "
        "retraction with appliance leakage and skin excoriation. PMH: Crohn "
        "disease, prior proctocolectomy 2020. Local revision only.",
        0, 0, 0, 0,
    ),

    # 323. Short endoscopic bladder neck procedure; spinal, day case.
    (
        "59M, ASA 2, for transurethral incision of the bladder neck. "
        "Indication: bladder neck contracture after prior TURP, obstructive "
        "voiding. PMH: prior TURP 2023, HTN.",
        0, 0, 0, 0,
    ),

    # 324. Catastrophic postpartum bleeding. Massive transfusion in a young
    #      woman; the kidney is what gives.
    (
        "31F, ASA 4E, for emergent peripartum hysterectomy. Indication: "
        "atonic postpartum hemorrhage unresponsive to uterotonics, balloon "
        "tamponade and B-Lynch suture, 5 L blood loss. PMH: none. 14 units "
        "transfused, DIC.",
        0, 0, 1, 0,
    ),

    # 325. Isolated patella fracture in a middle-aged patient; short, home the
    #      next day.
    (
        "50F, ASA 2, for tension band wiring of right patella. Indication: "
        "displaced transverse patella fracture with extensor lag after fall. "
        "PMH: hypothyroidism. Isolated injury.",
        0, 0, 0, 0,
    ),

    # 326. Twenty-minute pocket procedure under local in an elderly patient.
    (
        "84M, ASA 3, for pacemaker generator replacement. Indication: elective "
        "replacement indicator on pacemaker check, complete heart block, "
        "pacing-dependent. PMH: complete heart block, HTN, CKD stage 3. Local "
        "anesthesia.",
        0, 0, 0, 0,
    ),

    # 327. Anorectal surgery in an anticoagulated older patient; short, but she
    #      is well and goes home.
    (
        "74F, ASA 3, for stapled hemorrhoidopexy. Indication: grade III "
        "hemorrhoids with recurrent bleeding and symptomatic anemia. PMH: AF on "
        "warfarin, bridged perioperatively, HTN, hemoglobin 9.6.",
        0, 0, 0, 0,
    ),

    # 328. Brain abscess in an immunosuppressed patient; drained, but he has a
    #      parenchymal infection and a long antibiotic course.
    (
        "56M, ASA 4, for craniotomy with evacuation of cerebral abscess. "
        "Indication: 4 cm left temporal abscess with mass effect, from "
        "hematogenous spread. PMH: poorly controlled T2DM, IV drug use, "
        "endocarditis on treatment, seizure on admission.",
        0, 0, 0, 1,
    ),

    # 329. Chronic mesenteric ischemia in a cachectic smoker; the vessels and
    #      the nutrition are both poor.
    (
        "70F, ASA 4, for antegrade aortomesenteric bypass. Indication: chronic "
        "mesenteric ischemia with postprandial pain and 35 lb weight loss, "
        "failed stenting. PMH: current smoker, PVD, CKD stage 3, albumin 2.7, "
        "BMI 18.",
        0, 1, 1, 0,
    ),

    # 330. Short paediatric ENT case; healthy child.
    (
        "5F, ASA 1, for adenoidectomy with bilateral myringotomy and grommet "
        "insertion. Indication: obstructive sleep-disordered breathing with "
        "recurrent otitis media with effusion. No PMH.",
        0, 0, 0, 0,
    ),

    # 331. Liver abscess drained surgically after failed percutaneous drainage;
    #      he arrives septic.
    (
        "61M, ASA 4E, for laparoscopic drainage of pyogenic liver abscess. "
        "Indication: 9 cm right lobe abscess with sepsis, failed percutaneous "
        "drainage. PMH: T2DM HbA1c 10.1, recent cholangitis. Febrile 39.2, "
        "lactate 3.4.",
        0, 0, 1, 0,
    ),

    # 332. Partial penectomy in an older man; short perineal case.
    (
        "69M, ASA 3, for partial penectomy with bilateral inguinal sentinel "
        "node biopsy. Indication: T2 penile squamous cell carcinoma of the "
        "glans. PMH: phimosis, HTN, former smoker, T2DM.",
        1, 0, 0, 0,
    ),

    # 333. Adnexal surgery in the second trimester; brief and uneventful.
    (
        "29F, ASA 2, for laparoscopic ovarian cystectomy at 17 weeks gestation. "
        "Indication: persistent 9 cm ovarian cyst with torsion risk. PMH: "
        "uncomplicated pregnancy. Fetal viability confirmed postoperatively.",
        0, 0, 0, 0,
    ),

    # 334. Tibial plateau fracture in a middle-aged patient; non-weight-bearing
    #      for six weeks but otherwise well.
    (
        "52M, ASA 2, for open reduction internal fixation of left tibial "
        "plateau, Schatzker VI. Indication: bicondylar tibial plateau fracture "
        "after fall from height. PMH: HTN, BMI 32. Staged after external "
        "fixator.",
        1, 0, 0, 0,
    ),

    # 335. Total arch replacement with circulatory arrest; the most demanding
    #      cardiac operation there is.
    (
        "64M, ASA 4, for total aortic arch replacement with frozen elephant "
        "trunk. Indication: chronic type A dissection with arch aneurysm 6.2 "
        "cm, prior ascending repair. PMH: prior type A repair 2019, HTN, CKD "
        "stage 3. DHCA 52 min.",
        0, 1, 1, 1,
    ),

    # 336. Removing a failed band; short laparoscopic case in a mobile patient.
    (
        "42F, ASA 3, for laparoscopic removal of adjustable gastric band. "
        "Indication: band slippage with dysphagia and reflux. PMH: prior "
        "gastric band 2013, BMI 38, HTN. Uneventful, home day 1.",
        0, 0, 0, 0,
    ),

    # 337. Temporal lobectomy in a young epilepsy patient; extensively worked
    #      up, cognitively intact.
    (
        "33F, ASA 2, for left anterior temporal lobectomy with "
        "amygdalohippocampectomy. Indication: drug-resistant mesial temporal "
        "lobe epilepsy with hippocampal sclerosis. PMH: epilepsy 15 years, on "
        "three antiseizure medications. Wada and neuropsychology completed.",
        0, 0, 0, 0,
    ),

    # 338. Short line insertion in a dialysis patient; sedation, twenty minutes.
    (
        "55F, ASA 4, for tunnelled hemodialysis catheter insertion, right "
        "internal jugular. Indication: failed arteriovenous fistula with "
        "urgent dialysis requirement. PMH: ESRD from lupus nephritis, "
        "SLE on hydroxychloroquine, HTN.",
        0, 0, 0, 0,
    ),

    # 339. Elective nasal surgery in a healthy adult; day case.
    (
        "27F, ASA 1, for septorhinoplasty. Indication: post-traumatic nasal "
        "deformity with septal deviation and obstruction. No PMH. Nonsmoker.",
        0, 0, 0, 0,
    ),

    # 340. DISCORDANT vs row 89: appendicitis in an older patient, but early
    #      presentation, laparoscopic, not perforated, and normal kidneys.
    (
        "71F, ASA 2, for laparoscopic appendectomy. Indication: acute "
        "uncomplicated appendicitis, 12h of symptoms, no perforation on CT. "
        "PMH: hypothyroidism. Creatinine 0.8, independent, home day 1.",
        0, 0, 0, 0,
    ),

    # 341. Valve-in-valve for a degenerated bioprosthesis; avoids redo
    #      sternotomy in a frail woman, but the contrast load is real.
    (
        "86F, ASA 4, for valve-in-valve transcatheter aortic valve replacement. "
        "Indication: degenerated aortic bioprosthesis with severe stenosis, "
        "prohibitive redo risk. PMH: AVR 2012, CKD stage 3b Cr 2.0, frailty, "
        "AF, CHF.",
        0, 0, 1, 1,
    ),

    # 342. Perforated obstructing colon cancer. Faeculent contamination, cancer,
    #      and an elderly patient in shock.
    (
        "77M, ASA 5E, for emergent Hartmann procedure. Indication: perforated "
        "obstructing sigmoid adenocarcinoma with faeculent peritonitis. PMH: "
        "HTN, T2DM, CKD stage 3, former smoker. Hypotensive, lactate 4.9, on "
        "norepinephrine.",
        1, 1, 1, 1,
    ),

    # 343. Tumour thrombus in the cava: clamping above the renal veins with
    #      bypass standby.
    (
        "63M, ASA 4, for radical nephrectomy with level III inferior vena cava "
        "thrombectomy. Indication: right renal cell carcinoma with tumour "
        "thrombus to the intrahepatic cava. PMH: HTN, BMI 30. Anticipated 6h, "
        "EBL 2500 mL.",
        1, 0, 1, 0,
    ),

    # 344. Hysterectomy in a superobese patient; long case, difficult access,
    #      but she is mobile afterwards.
    (
        "51F, ASA 3, for total abdominal hysterectomy with bilateral "
        "salpingo-oophorectomy. Indication: symptomatic fibroid uterus, 22 "
        "weeks size, with menorrhagia. PMH: BMI 52, T2DM, OSA on CPAP, HTN.",
        1, 0, 0, 0,
    ),

    # 345. Periprosthetic fracture in an elderly patient; long revision case
    #      and weeks of restricted weight-bearing.
    (
        "83F, ASA 4, for open reduction internal fixation of periprosthetic "
        "femoral fracture with revision stem. Indication: Vancouver B2 fracture "
        "around a loose hip stem after fall. PMH: THA 2011, osteoporosis, mild "
        "dementia, HTN, CKD stage 3.",
        1, 0, 0, 1,
    ),

    # 346. Spinal trauma in a young man; instrumented same day, mobilised in a
    #      brace.
    (
        "36M, ASA 3E, for T12 posterior instrumented fusion. Indication: "
        "unstable burst fracture after fall from height, no neurological "
        "deficit. PMH: none. Isolated spinal injury.",
        0, 0, 0, 0,
    ),

    # 347. Crossover bypass to salvage a leg; short extra-anatomic case under
    #      regional.
    (
        "72M, ASA 4, for right-to-left femorofemoral crossover bypass. "
        "Indication: left iliac occlusion with critical limb ischemia, unfit "
        "for aortic surgery. PMH: COPD, CAD, CKD stage 3, former smoker 50 "
        "pack-years. Epidural anesthesia.",
        0, 0, 0, 0,
    ),

    # 348. Salvage neck dissection in a radiated field; long, fibrotic, and his
    #      swallow was already poor.
    (
        "62M, ASA 3, for salvage modified radical neck dissection. Indication: "
        "recurrent nodal disease 14 months after chemoradiation for "
        "oropharyngeal carcinoma. PMH: prior chemoradiation, current smoker, "
        "chronic dysphagia, PEG-dependent.",
        0, 1, 0, 0,
    ),

    # 349. Pressure sore reconstruction in a paraplegic patient; immobile by
    #      definition, prone flap, six weeks off the flap afterwards.
    (
        "51M, ASA 4, for gluteal rotation flap coverage of sacral pressure "
        "ulcer. Indication: stage IV sacral ulcer with exposed bone after "
        "debridement. PMH: T6 paraplegia 15 years, wheelchair user, recurrent "
        "UTI, prior DVT, anemia.",
        1, 0, 0, 0,
    ),

    # 350. Diaphragm plication in a dyspnoeic patient; the diaphragm is the
    #      problem and the lung underneath it is chronically compressed.
    (
        "59M, ASA 3, for thoracoscopic diaphragm plication. Indication: right "
        "hemidiaphragm paralysis with orthopnea after phrenic nerve injury. "
        "PMH: prior cardiac surgery 2023, obesity BMI 37, OSA.",
        0, 1, 0, 0,
    ),

    # 351. Exsanguinating upper GI bleed in an elderly patient; oversewn, but
    #      he arrives in shock after litres of blood.
    (
        "76M, ASA 5E, for emergent laparotomy with oversewing of bleeding "
        "duodenal ulcer. Indication: massive upper GI hemorrhage, failed "
        "endoscopic therapy twice, hemoglobin 5.2. PMH: NSAID use, CKD stage 3, "
        "CAD, AF. 10 units transfused.",
        1, 1, 1, 1,
    ),

    # 352. Highly sensitised recipient with desensitisation; heavy
    #      immunosuppression and a graft that takes days to work.
    (
        "46F, ASA 4, for deceased donor kidney transplant with plasmapheresis "
        "and IVIG desensitisation. Indication: ESRD, cPRA 98%, waited 8 years. "
        "PMH: ESRD from lupus nephritis, three prior pregnancies, prior failed "
        "graft. Cold ischemia 19h.",
        0, 0, 1, 0,
    ),

    # 353. Circumferential burns; escharotomy is minutes long but the
    #      inhalation injury is already established.
    (
        "44M, ASA 4E, for bilateral upper limb and chest escharotomy. "
        "Indication: circumferential full-thickness burns with compartment "
        "syndrome and restricted ventilation, 45% TBSA. Intubated for "
        "inhalation injury, carboxyhemoglobin 24%.",
        1, 1, 1, 0,
    ),

    # 354. Palliative gastrectomy in a cachectic patient with advanced disease.
    (
        "71M, ASA 4, for palliative distal gastrectomy. Indication: bleeding "
        "unresectable gastric adenocarcinoma with transfusion dependence. PMH: "
        "metastatic gastric cancer, albumin 2.3, 40 lb weight loss, COPD, "
        "former smoker.",
        1, 1, 0, 0,
    ),

    # 355. Open prostatectomy in a well patient; more blood loss than robotic
    #      but he is fit.
    (
        "58M, ASA 2, for open retropubic radical prostatectomy with pelvic "
        "lymphadenectomy. Indication: high-risk prostate adenocarcinoma, "
        "Gleason 4+4, PSA 22. PMH: none. Nonsmoker, BMI 25.",
        1, 0, 0, 0,
    ),

    # 356. Large fibroid uterus in a well woman; open, bloody, but she is 46
    #      and mobile the next day.
    (
        "46F, ASA 2, for total abdominal hysterectomy. Indication: fibroid "
        "uterus 18 weeks size with pressure symptoms and anemia. PMH: iron "
        "deficiency anemia hemoglobin 8.9, BMI 29. Two units transfused.",
        0, 0, 0, 0,
    ),

    # 357. Charcot reconstruction in a diabetic; long case, non-weight-bearing
    #      for months, poor tissue.
    (
        "60M, ASA 4, for tibiotalocalcaneal fusion with intramedullary nail. "
        "Indication: Charcot neuroarthropathy with unstable hindfoot and "
        "recurrent ulceration. PMH: T2DM HbA1c 9.6, peripheral neuropathy, CKD "
        "stage 3, PVD, obesity.",
        1, 0, 0, 0,
    ),

    # 358. Robotic mitral repair in a fit patient; short, small incisions, home
    #      on day 4.
    (
        "52M, ASA 3, for robotic mitral valve repair with annuloplasty ring. "
        "Indication: severe degenerative mitral regurgitation, P2 prolapse. "
        "PMH: hyperlipidemia. EF 62%, normal creatinine, never smoker.",
        0, 0, 0, 0,
    ),

    # 359. DISCORDANT vs row 23: same operation, but 34 years old with an
    #      isolated traumatic bleed and a brain that recovers.
    (
        "34M, ASA 4E, for emergent craniotomy with evacuation of acute subdural "
        "hematoma. Indication: traumatic acute SDH after assault, GCS 13, "
        "midline shift 7 mm. No PMH. Not anticoagulated.",
        0, 0, 0, 0,
    ),

    # 360. Fenestrated endograft with renal stenting; a big dye load delivered
    #      directly across the renal ostia.
    (
        "74M, ASA 4, for fenestrated endovascular aneurysm repair with bilateral "
        "renal stenting. Indication: 6.1 cm juxtarenal abdominal aortic "
        "aneurysm. PMH: CKD stage 3 Cr 1.9, COPD, HTN, former smoker. Contrast "
        "180 mL, 4h fluoroscopy.",
        0, 0, 1, 0,
    ),

    # 361. Short airway dilatation in a young patient; ten minutes, jet
    #      ventilation.
    (
        "26F, ASA 3, for microlaryngoscopy with balloon dilatation of subglottic "
        "stenosis. Indication: idiopathic subglottic stenosis with exertional "
        "stridor, third dilatation. PMH: idiopathic subglottic stenosis, "
        "otherwise well.",
        0, 0, 0, 0,
    ),

    # 362. Adult intussusception with a lead point; resected, uneventful.
    (
        "54M, ASA 2E, for laparoscopic right hemicolectomy. Indication: "
        "ileocolic intussusception with a lipoma lead point on CT, subacute "
        "obstruction. PMH: none. No perforation.",
        0, 0, 0, 0,
    ),

    # 363. Brief cystoscopic biopsy in an elderly woman with dementia; short,
    #      but the anesthetic and the ward are enough to tip her.
    (
        "87F, ASA 4, for cystoscopy with bladder biopsy under general "
        "anesthesia. Indication: painless hematuria with a suspicious bladder "
        "lesion. PMH: moderate dementia, AF on apixaban, CKD stage 3, prior "
        "delirium after previous admission.",
        0, 0, 0, 1,
    ),

    # 364. Young woman, laparoscopic, short case.
    (
        "31F, ASA 2, for laparoscopic salpingostomy. Indication: persistent "
        "tubal ectopic pregnancy after failed methotrexate, beta-hCG plateau, "
        "hemodynamically stable. PMH: prior ectopic 2021, desires fertility.",
        0, 0, 0, 0,
    ),

    # 365. Hip fracture in a man with severe COPD. Both the archetypes collide.
    (
        "82M, ASA 4, for left hip intramedullary nailing. Indication: "
        "intertrochanteric fracture after fall at home. PMH: severe COPD FEV1 "
        "35%, home O2, current smoker, prior pneumonia x3, HTN. Uses walker.",
        0, 1, 0, 1,
    ),

    # 366. Endocarditis with embolic phenomena; vegetation, nephrotoxic
    #      antibiotics, and septic emboli to the brain.
    (
        "38M, ASA 4E, for emergent mitral valve replacement. Indication: acute "
        "mitral endocarditis with 18 mm vegetation, cerebral septic emboli and "
        "heart failure. PMH: IV drug use, hepatitis C. On vancomycin and "
        "gentamicin, EF 45%.",
        0, 1, 1, 1,
    ),

    # 367. Elective colon cancer resection in a fit patient; laparoscopic,
    #      short, home day 3.
    (
        "57M, ASA 2, for laparoscopic extended right hemicolectomy. Indication: "
        "splenic flexure adenocarcinoma, T3N0. PMH: hyperlipidemia, BMI 27. "
        "Never smoker, enhanced recovery pathway.",
        1, 0, 0, 0,
    ),

    # 368. Short device revision under sedation.
    (
        "49M, ASA 3, for spinal cord stimulator lead revision. Indication: lead "
        "migration with loss of coverage, 8 months after implantation. PMH: "
        "chronic pain, prior lumbar fusion, depression. Sedation only.",
        0, 0, 0, 0,
    ),

    # 369. Groin revascularisation in a diabetic; short, regional, limb
    #      salvaged.
    (
        "68M, ASA 4, for left common femoral endarterectomy with profundaplasty "
        "and bovine patch. Indication: common femoral occlusive disease with "
        "rest pain. PMH: T2DM, PVD, CAD s/p CABG, current smoker, CKD stage 3.",
        0, 0, 0, 0,
    ),

    # 370. Middle ear reconstruction in a healthy adult; day case.
    (
        "36F, ASA 1, for right tympanoplasty with cartilage graft. Indication: "
        "persistent tympanic membrane perforation with conductive hearing loss "
        "after chronic otitis. No PMH.",
        0, 0, 0, 0,
    ),

    # 371. Pseudocyst drainage in a patient with a long pancreatitis course;
    #      already deconditioned but the operation goes well.
    (
        "47M, ASA 3, for open cystgastrostomy. Indication: symptomatic 12 cm "
        "pancreatic pseudocyst with gastric outlet compression, 10 weeks after "
        "alcoholic pancreatitis. PMH: alcohol use disorder, abstinent 10 weeks, "
        "new T2DM.",
        0, 0, 0, 0,
    ),

    # 372. Prostate resection in a man on antiplatelets after recent stenting;
    #      bloody, but he is well and goes home.
    (
        "71M, ASA 3, for transurethral resection of prostate. Indication: "
        "refractory retention with recurrent hematuria. PMH: recent drug-eluting "
        "stent 8 months ago, on clopidogrel continued perioperatively, HTN, "
        "hyperlipidemia. Independent.",
        0, 0, 0, 0,
    ),

    # 373. Short gynecologic day case in a well patient.
    (
        "38F, ASA 2, for large loop excision of the transformation zone under "
        "general anesthesia. Indication: CIN 3 on colposcopic biopsy, "
        "unsuitable for local anesthesia. PMH: none. Day case.",
        0, 0, 0, 0,
    ),

    # 374. Arthroscopic shoulder surgery in a healthy adult; beach chair,
    #      block, day case.
    (
        "44M, ASA 2, for arthroscopic Bankart repair. Indication: recurrent "
        "anterior shoulder instability, four dislocations. PMH: HTN. "
        "Interscalene block, day case.",
        0, 0, 0, 0,
    ),

    # 375. Bypass surgery in a current smoker with COPD; the sternotomy and the
    #      chronic lung disease compound each other.
    (
        "66M, ASA 4, for coronary artery bypass grafting x4 with left internal "
        "mammary artery. Indication: three-vessel disease with reduced EF 40%. "
        "PMH: COPD FEV1 48%, current smoker 40 pack-years, T2DM, HTN, CKD stage "
        "2.",
        0, 1, 0, 0,
    ),

    # 376. Perineal approach in a frail elderly woman; short, spinal, avoids
    #      the abdomen.
    (
        "88F, ASA 4, for Altemeier perineal rectosigmoidectomy. Indication: "
        "full-thickness rectal prolapse with incontinence, unfit for abdominal "
        "approach. PMH: mild dementia, CHF, AF, CKD stage 3. Spinal anesthesia, "
        "assisted living.",
        0, 0, 0, 1,
    ),

    # 377. Nerve exploration and grafting in a young man; long but
    #      physiologically light.
    (
        "24M, ASA 2, for brachial plexus exploration with sural nerve grafting. "
        "Indication: upper trunk brachial plexus injury after motorcycle "
        "collision 5 months ago, no recovery. PMH: prior clavicle fracture. "
        "Otherwise well.",
        0, 0, 0, 0,
    ),

    # 378. Iliac endarterectomy in a smoker; retroperitoneal approach, but no
    #      suprarenal clamp and he does well.
    (
        "61M, ASA 3, for left iliac endarterectomy with patch angioplasty. "
        "Indication: focal common iliac stenosis with disabling claudication, "
        "failed angioplasty. PMH: current smoker 30 pack-years, HTN, "
        "hyperlipidemia. Creatinine 1.0.",
        0, 0, 0, 0,
    ),

    # 379. Endoscopic laser resection of an early laryngeal cancer; short,
    #      preserves the airway.
    (
        "58M, ASA 3, for transoral laser cordectomy. Indication: T1a glottic "
        "squamous cell carcinoma, hoarseness. PMH: former smoker 30 pack-years, "
        "quit 2 years ago, HTN. Airway patent, no prior radiotherapy.",
        0, 0, 0, 0,
    ),

    # 380. Emergency hernia surgery in a cirrhotic with ascites; the liver
    #      drives everything.
    (
        "56M, ASA 4E, for emergent umbilical hernia repair with small bowel "
        "resection. Indication: incarcerated umbilical hernia through ascitic "
        "abdominal wall with ischemic bowel. PMH: alcoholic cirrhosis "
        "Child-Pugh C, refractory ascites, coagulopathy INR 1.9, "
        "thrombocytopenia.",
        0, 1, 1, 1,
    ),

    # 381. Cecal volvulus in a young adult; resected, viable, home in four days.
    (
        "38F, ASA 2E, for emergent laparoscopic right hemicolectomy. "
        "Indication: cecal volvulus with dilated cecum on CT, no ischemia at "
        "operation. PMH: none. Presented within 12h.",
        0, 0, 0, 0,
    ),

    # 382. Adult coarctation repair; young patient, left thoracotomy, clean
    #      course.
    (
        "29M, ASA 3, for repair of aortic coarctation with interposition graft "
        "via left thoracotomy. Indication: native coarctation with refractory "
        "hypertension and 40 mmHg gradient. PMH: bicuspid aortic valve, HTN on "
        "three agents.",
        0, 0, 0, 0,
    ),

    # 383. Nephron-sparing surgery in a patient whose kidneys are already
    #      compromised; the clamp is what costs him.
    (
        "66M, ASA 3, for robotic partial nephrectomy. Indication: 4.2 cm renal "
        "mass, endophytic, RENAL score 10. PMH: CKD stage 3b Cr 2.0, T2DM, HTN, "
        "obesity BMI 35. Warm ischemia 31 min.",
        0, 0, 1, 0,
    ),

    # 384. Elective microsurgery in a healthy young woman; long but light.
    (
        "35F, ASA 1, for laparoscopic tubal reanastomosis. Indication: desired "
        "fertility after tubal ligation, patent proximal and distal segments. "
        "No PMH. Otherwise well.",
        0, 0, 0, 0,
    ),

    # 385. Hip fracture in a Parkinson patient; rigidity, poor cough, and
    #      medication timing disrupted by fasting.
    (
        "80M, ASA 4, for right hip hemiarthroplasty. Indication: displaced "
        "femoral neck fracture after freezing episode and fall. PMH: Parkinson "
        "disease 11 years, on levodopa four-hourly, dysphagia, prior aspiration "
        "pneumonia, orthostatic hypotension.",
        0, 1, 0, 1,
    ),

    # 386. Anterior cervical corpectomy in a myelopathic patient; airway
    #      swelling and a difficult swallow afterwards.
    (
        "69M, ASA 3, for C5 corpectomy with strut graft and anterior plating. "
        "Indication: cervical myelopathy with cord signal change and progressive "
        "weakness. PMH: HTN, former smoker, OSA untreated, BMI 34.",
        0, 1, 0, 0,
    ),

    # 387. Endograft for a complicated dissection; malperfusion, spinal drain,
    #      contrast, and an already-injured kidney.
    (
        "58M, ASA 4E, for thoracic endovascular aortic repair. Indication: "
        "complicated type B aortic dissection with renal malperfusion and "
        "refractory hypertension. PMH: uncontrolled HTN, current smoker. "
        "Creatinine risen 1.1 to 2.4 over 48h.",
        0, 0, 1, 0,
    ),

    # 388. Salvage laryngectomy in a radiated neck; poor healing, permanent
    #      stoma, chronically aspirating patient.
    (
        "64M, ASA 4, for salvage total laryngectomy. Indication: recurrent "
        "glottic carcinoma 18 months after definitive chemoradiation. PMH: prior "
        "chemoradiation, current smoker, COPD, malnutrition albumin 2.9, "
        "PEG-dependent.",
        0, 1, 0, 0,
    ),

    # 389. Pedicled flap reconstruction; long case with the abdomen and chest
    #      both operated, in a patient with active malignancy.
    (
        "56F, ASA 3, for delayed breast reconstruction with pedicled TRAM flap. "
        "Indication: reconstruction 14 months after mastectomy and radiotherapy "
        "for invasive ductal carcinoma. PMH: breast cancer on anastrozole, BMI "
        "34, HTN, former smoker.",
        1, 0, 0, 0,
    ),

    # 390. Chest wall correction in a healthy teenager; painful but
    #      physiologically trivial.
    (
        "17M, ASA 1, for Nuss repair of pectus excavatum. Indication: severe "
        "pectus excavatum, Haller index 4.2, with exertional dyspnea. No PMH. "
        "Thoracic epidural for analgesia.",
        0, 0, 0, 0,
    ),

    # 391. Staple line leak after bariatric surgery; septic, reoperated, but
    #      young and mobile.
    (
        "39F, ASA 4E, for emergent laparoscopic washout with drainage and "
        "feeding jejunostomy. Indication: staple line leak day 5 after sleeve "
        "gastrectomy, tachycardic and febrile. PMH: recent sleeve, BMI 46, OSA, "
        "T2DM.",
        0, 1, 0, 0,
    ),

    # 392. Living donor liver transplant; better graft, shorter ischemia, but
    #      still a MELD 28 recipient.
    (
        "51M, ASA 4, for living donor right lobe liver transplant. Indication: "
        "decompensated hepatitis B cirrhosis, MELD 28, with hepatocellular "
        "carcinoma within Milan criteria. PMH: hepatitis B cirrhosis, portal "
        "hypertension, prior variceal bleed.",
        0, 0, 1, 1,
    ),

    # 393. Preperitoneal packing for a bleeding pelvis; young, resuscitated,
    #      recovers.
    (
        "31M, ASA 5E, for preperitoneal pelvic packing with external fixation. "
        "Indication: hemodynamically unstable open-book pelvic fracture after "
        "motorcycle collision, transient responder. No PMH. 12 units "
        "transfused, angioembolisation afterwards.",
        1, 0, 1, 0,
    ),

    # 394. Colon cancer resection in a nonagenarian; open, cancer, and a long
    #      hospital stay away from home.
    (
        "91M, ASA 4, for open right hemicolectomy. Indication: symptomatic "
        "ascending colon adenocarcinoma with anemia and weight loss. PMH: mild "
        "cognitive impairment, AF on apixaban, CKD stage 3, HTN. Lives with "
        "daughter.",
        1, 1, 0, 1,
    ),

    # 395. Cystectomy in an elderly man; long case, urinary diversion, and a
    #      frail host.
    (
        "80M, ASA 4, for radical cystectomy with ileal conduit. Indication: "
        "muscle-invasive bladder cancer, unfit for neoadjuvant chemotherapy. "
        "PMH: former smoker 50 pack-years, CKD stage 3, CAD, mild cognitive "
        "impairment, sarcopenia.",
        1, 1, 1, 1,
    ),

    # 396. Endometrial cancer surgery in a frail elderly woman; robotic and
    #      short, but she is 85 with cognitive decline.
    (
        "85F, ASA 4, for robotic hysterectomy with bilateral "
        "salpingo-oophorectomy. Indication: grade 1 endometrioid adenocarcinoma "
        "with postmenopausal bleeding. PMH: mild dementia, CHF, T2DM, BMI 38, "
        "CKD stage 3.",
        1, 0, 0, 1,
    ),

    # 397. Elective hip in a Parkinson patient; rigidity and postural
    #      instability, but no dementia and a short spinal case.
    (
        "71M, ASA 3, for left total hip arthroplasty. Indication: end-stage "
        "osteoarthritis with night pain. PMH: Parkinson disease 5 years, well "
        "controlled on levodopa, HTN. Cognitively intact, walks unaided.",
        0, 0, 0, 1,
    ),

    # 398. Atrial myxoma excision in a fit patient; short pump run, curative.
    (
        "48F, ASA 3, for excision of left atrial myxoma. Indication: 5 cm "
        "pedunculated left atrial myxoma with syncope and embolic TIA. PMH: "
        "recent TIA, otherwise well. EF 60%, normal creatinine.",
        0, 0, 0, 0,
    ),

    # 399. Skull base surgery: hours long, cranial nerves at risk, but she is
    #      54 and cognitively normal.
    (
        "54F, ASA 3, for retrosigmoid craniotomy for resection of petroclival "
        "meningioma. Indication: 4 cm skull base meningioma with trigeminal "
        "compression and brainstem distortion. PMH: HTN. Anticipated 9h case.",
        0, 0, 0, 1,
    ),

    # 400. Trivial diagnostic procedure under local.
    (
        "73F, ASA 3, for right temporal artery biopsy. Indication: suspected "
        "giant cell arteritis with jaw claudication and ESR 92, on high-dose "
        "prednisolone. PMH: polymyalgia rheumatica, HTN. Local anesthesia.",
        0, 0, 0, 0,
    ),

    # 401. Appendicitis in an obese patient; laparoscopic, uneventful.
    (
        "41M, ASA 3E, for laparoscopic appendectomy. Indication: acute "
        "appendicitis, 20h of symptoms. PMH: BMI 44, T2DM, OSA on CPAP, HTN. "
        "Not perforated, home day 1.",
        0, 0, 0, 0,
    ),

    # 402. Short paediatric case in a healthy boy.
    (
        "3M, ASA 1, for right orchidopexy. Indication: palpable undescended "
        "right testis in the inguinal canal. No PMH. Caudal block, day case.",
        0, 0, 0, 0,
    ),

    # 403. Planned repeat cesarean in a healthy woman; spinal, forty minutes.
    (
        "34F, ASA 2, for elective repeat cesarean section at 39 weeks. "
        "Indication: two prior cesareans, declines trial of labour. PMH: two "
        "prior cesareans, otherwise well. Spinal anesthesia.",
        0, 0, 0, 0,
    ),

    # 404. Small foot fracture in a healthy adult; day case.
    (
        "29F, ASA 1, for open reduction internal fixation of fifth metatarsal. "
        "Indication: displaced Jones fracture in a competitive runner. No PMH. "
        "Ankle block, day case.",
        0, 0, 0, 0,
    ),

    # 405. Peripheral nerve tumour excision; short, regional.
    (
        "43M, ASA 2, for excision of schwannoma from the right median nerve at "
        "the forearm. Indication: enlarging painful mass with Tinel sign. PMH: "
        "none. Supraclavicular block.",
        0, 0, 0, 0,
    ),

    # 406. Adult VSD closure in a young patient; short pump run.
    (
        "27F, ASA 3, for surgical closure of perimembranous ventricular septal "
        "defect. Indication: significant left-to-right shunt with Qp:Qs 2.1 and "
        "LV volume overload. PMH: congenital VSD. Pulmonary pressures normal.",
        0, 0, 0, 0,
    ),

    # 407. DISCORDANT vs rows 42 and 52: severe COPD, but a groin hernia under
    #      regional with no abdominal incision and no chest insult.
    (
        "75M, ASA 4, for open left inguinal hernia repair with mesh. "
        "Indication: symptomatic inguinal hernia, enlarging. PMH: severe COPD "
        "FEV1 38%, home O2 nocturnal, former smoker 50 pack-years. Local "
        "infiltration with sedation, day case.",
        0, 0, 0, 0,
    ),

    # 408. Lung resection in interstitial disease; the parenchyma is already
    #      fibrotic and does not tolerate the insult.
    (
        "68M, ASA 4, for VATS left lower lobectomy. Indication: 3 cm "
        "adenocarcinoma in a background of idiopathic pulmonary fibrosis. PMH: "
        "IPF on antifibrotics, DLCO 42% predicted, former smoker, home O2 on "
        "exertion.",
        0, 1, 0, 0,
    ),

    # 409. Return to theatre for a bleeding tonsil bed; short, but the swallowed
    #      blood and the emergency airway are the risk.
    (
        "19M, ASA 2E, for emergent arrest of post-tonsillectomy hemorrhage. "
        "Indication: secondary bleed day 7 after tonsillectomy, ongoing "
        "bleeding, hemoglobin 10.1. PMH: recent tonsillectomy. Rapid sequence "
        "induction, large volume of swallowed blood.",
        0, 0, 0, 0,
    ),

    # 410. Emergency colectomy in a young patient on high-dose steroids and
    #      biologics; toxic colitis but a resilient host.
    (
        "27M, ASA 4E, for emergent subtotal colectomy with end ileostomy. "
        "Indication: acute severe ulcerative colitis with toxic dilation, "
        "failed rescue infliximab. PMH: UC 3 years, on prednisolone 60 mg, "
        "albumin 2.5, CRP 180.",
        0, 0, 0, 0,
    ),

    # 411. Ureteric reimplantation in a young adult; short, clean.
    (
        "32F, ASA 2, for laparoscopic ureteroneocystostomy with psoas hitch. "
        "Indication: distal ureteric stricture after prior ureteroscopy, "
        "obstructive on renography. PMH: nephrolithiasis, normal contralateral "
        "kidney.",
        0, 0, 0, 0,
    ),

    # 412. Short hysteroscopic procedure in a well woman.
    (
        "41F, ASA 2, for hysteroscopic resection of submucosal fibroid. "
        "Indication: type 1 submucosal fibroid with menorrhagia and "
        "infertility. PMH: iron deficiency anemia. Day case, LMA.",
        0, 0, 0, 0,
    ),

    # 413. Minor wrist arthroscopy in a healthy adult.
    (
        "34M, ASA 1, for right wrist arthroscopy with triangular fibrocartilage "
        "complex debridement. Indication: ulnar-sided wrist pain after fall, "
        "TFCC tear on MRI. No PMH. Regional block, day case.",
        0, 0, 0, 0,
    ),

    # 414. Device explant after myocardial recovery; redo sternotomy, but he is
    #      young and his ventricle has come back.
    (
        "42M, ASA 4, for left ventricular assist device explantation. "
        "Indication: myocardial recovery after 18 months of support, EF "
        "improved to 52% on turn-down study. PMH: peripartum-pattern "
        "nonischemic cardiomyopathy, LVAD 2024.",
        0, 0, 0, 0,
    ),

    # 415. Gallstone ileus in a frail elderly woman; late presentation, dilated
    #      bowel, and a long delay before diagnosis.
    (
        "85F, ASA 4E, for emergent laparotomy with enterolithotomy. Indication: "
        "gallstone ileus with 4 cm stone impacted in terminal ileum, 5 days of "
        "vomiting. PMH: chronic cholecystitis, mild dementia, CKD stage 3, HTN, "
        "AF.",
        0, 1, 1, 1,
    ),

    # 416. Microvascular decompression in a fit patient; posterior fossa but
    #      short and clean.
    (
        "57F, ASA 2, for microvascular decompression of the trigeminal nerve. "
        "Indication: classical trigeminal neuralgia with vascular loop on MRI, "
        "refractory to carbamazepine. PMH: none. Cognitively intact.",
        0, 0, 0, 0,
    ),

    # 417. Transplant renal artery repair; the graft is the only kidney and it
    #      is clamped.
    (
        "44M, ASA 4, for open repair of transplant renal artery stenosis with "
        "interposition graft. Indication: refractory hypertension and rising "
        "creatinine, failed angioplasty twice. PMH: kidney transplant 2021, "
        "baseline Cr 1.8, on tacrolimus, HTN.",
        0, 0, 1, 0,
    ),

    # 418. Deep lobe parotid surgery; longer than a superficial, but a well
    #      patient.
    (
        "55F, ASA 2, for total parotidectomy with facial nerve preservation. "
        "Indication: deep lobe pleomorphic adenoma, 5 cm, with parapharyngeal "
        "extension. PMH: hypothyroidism. Otherwise well.",
        0, 0, 0, 0,
    ),

    # 419. Gallbladder surgery in a patient with obesity hypoventilation; the
    #      chest is the problem, not the gallbladder.
    (
        "50M, ASA 4, for laparoscopic cholecystectomy. Indication: acute "
        "cholecystitis. PMH: BMI 61, obesity hypoventilation syndrome, home "
        "BiPAP, pulmonary hypertension, T2DM, baseline PaCO2 54.",
        0, 1, 0, 0,
    ),

    # 420. DISCORDANT vs rows 46 and 189: revision arthroplasty, but a fit
    #      64-year-old, short case, and mobilised the same day.
    (
        "64F, ASA 2, for revision of right total knee arthroplasty, polyethylene "
        "exchange only. Indication: isolated polyethylene wear with effusion, "
        "components well fixed. PMH: hypothyroidism, BMI 26. Mobilised day 0.",
        0, 0, 0, 0,
    ),

    # 421. Reoperation for bleeding on the first night; back to theatre, but
    #      the problem is mechanical and he is fit.
    (
        "59M, ASA 3E, for emergent relaparotomy for hemorrhage. Indication: "
        "bleeding from mesenteric vessel 8h after laparoscopic sigmoid "
        "colectomy, hemoglobin fall to 6.8. PMH: sigmoid cancer, HTN. Four "
        "units transfused.",
        0, 0, 0, 0,
    ),

    # 422. Structural valve procedure in a dialysis patient; already anuric, so
    #      the contrast has nothing left to injure.
    (
        "75M, ASA 4, for transcatheter aortic valve replacement, transfemoral. "
        "Indication: severe aortic stenosis with heart failure. PMH: ESRD on "
        "hemodialysis 5 years, T2DM, PVD, prior MI, frailty.",
        0, 0, 0, 1,
    ),

    # 423. Bladder perforation during resection; converted to open repair, but
    #      he is well and recovers.
    (
        "67M, ASA 3E, for transurethral resection of bladder tumour with open "
        "repair of extraperitoneal perforation. Indication: large lateral wall "
        "tumour with obturator jerk and perforation. PMH: HTN, former smoker. "
        "Catheter for 10 days.",
        0, 0, 0, 0,
    ),

    # 424. Hysterectomy in an anticoagulated woman; bleeding risk managed, but
    #      the operation itself is routine.
    (
        "54F, ASA 3, for total laparoscopic hysterectomy. Indication: "
        "symptomatic fibroids with menorrhagia on anticoagulation. PMH: prior "
        "pulmonary embolism 2022 on apixaban, held preoperatively, BMI 33, "
        "HTN.",
        1, 0, 0, 0,
    ),

    # 425. Hip fracture in a stroke survivor; hemiparesis, dysphagia, and a
    #      brain with no reserve.
    (
        "78F, ASA 4, for left hip hemiarthroplasty. Indication: displaced "
        "femoral neck fracture after fall on the hemiplegic side. PMH: prior "
        "left MCA infarct with right hemiparesis and dysphagia, AF on apixaban, "
        "HTN, vascular cognitive impairment.",
        0, 1, 0, 1,
    ),

    # 426. Ruptured AVM in a young patient; parenchymal bleed, but she is 28
    #      and recovers well.
    (
        "28F, ASA 4E, for emergent craniotomy with hematoma evacuation and AVM "
        "resection. Indication: ruptured left parietal AVM with 40 mL "
        "intraparenchymal hemorrhage, GCS 13. PMH: none.",
        1, 0, 0, 1,
    ),

    # 427. Amputation in a young diabetic; devastating, but he is 38 and
    #      physiologically robust.
    (
        "38M, ASA 3, for right below-knee amputation. Indication: chronic "
        "nonhealing plantar ulcer with calcaneal osteomyelitis, failed "
        "revascularisation. PMH: T1DM 26 years, peripheral neuropathy, prior "
        "contralateral toe amputations. Independent.",
        0, 0, 0, 0,
    ),

    # 428. Thyroid surgery in the second trimester; short, uneventful.
    (
        "31F, ASA 2, for right thyroid lobectomy at 18 weeks gestation. "
        "Indication: rapidly enlarging nodule with suspicious cytology, "
        "Bethesda V. PMH: uncomplicated pregnancy, euthyroid.",
        0, 0, 0, 0,
    ),

    # 429. Digital replantation in a young man; long microsurgery, but light on
    #      the rest of him.
    (
        "26M, ASA 2E, for replantation of right thumb. Indication: complete "
        "amputation at the level of the proximal phalanx from a table saw, "
        "ischemia time 4h. No PMH. Supraclavicular catheter for sympathetic "
        "block.",
        0, 0, 0, 0,
    ),

    # 430. DISCORDANT vs row 63: same disease, but robotic and no sternotomy,
    #      and she is optimised on IVIG beforehand.
    (
        "38F, ASA 3, for robotic thymectomy. Indication: generalised myasthenia "
        "gravis, thymoma 3 cm. PMH: myasthenia on pyridostigmine, IVIG course "
        "completed preoperatively, no bulbar symptoms. Extubated on table.",
        0, 0, 0, 0,
    ),

    # 431. Elective hernia in a compensated cirrhotic; ascites controlled, and
    #      he does well.
    (
        "52M, ASA 3, for open umbilical hernia repair with mesh. Indication: "
        "symptomatic umbilical hernia with skin thinning, ascites controlled on "
        "diuretics. PMH: NASH cirrhosis Child-Pugh A, T2DM, BMI 34. Elective, "
        "optimised.",
        0, 0, 0, 0,
    ),

    # 432. Transplant in an obese recipient; technically difficult, wound
    #      issues, and delayed function.
    (
        "49M, ASA 4, for deceased donor kidney transplant. Indication: ESRD on "
        "hemodialysis 4 years, diabetic nephropathy. PMH: T2DM, BMI 41, HTN, "
        "OSA on CPAP. Cold ischemia 17h, deep tissues, long incision.",
        0, 0, 1, 0,
    ),

    # 433. Penetrating neck injury in a young man; explored, repaired,
    #      uneventful.
    (
        "23M, ASA 4E, for emergent neck exploration with repair of internal "
        "jugular vein. Indication: zone II stab wound with expanding hematoma "
        "and airway deviation. No PMH. Intubated in the trauma bay.",
        0, 0, 0, 0,
    ),

    # 434. Ischemic colitis after a low-flow state; the kidneys were injured by
    #      the same event that killed the colon.
    (
        "74F, ASA 5E, for emergent subtotal colectomy with end ileostomy. "
        "Indication: ischemic colitis with perforation after prolonged "
        "hypotension during cardiac arrest resuscitation. PMH: CAD, CHF, CKD "
        "stage 3, AF. On vasopressors, lactate 6.2.",
        1, 1, 1, 1,
    ),

    # 435. Retroperitoneal node dissection in a young man after chemotherapy;
    #      long case but a resilient host.
    (
        "31M, ASA 3, for post-chemotherapy retroperitoneal lymph node "
        "dissection. Indication: residual 4 cm retroperitoneal mass after BEP "
        "chemotherapy for nonseminomatous germ cell tumour. PMH: testicular "
        "cancer, bleomycin exposure. Anticipated 5h case.",
        1, 0, 0, 0,
    ),

    # 436. Adnexal torsion in an older woman; short laparoscopic case,
    #      uneventful.
    (
        "66F, ASA 3, for emergent laparoscopic bilateral salpingo-oophorectomy. "
        "Indication: right ovarian torsion around a 7 cm cyst, acute pain. PMH: "
        "HTN, T2DM, BMI 32. Cyst benign on frozen section.",
        0, 0, 0, 0,
    ),

    # 437. Arthroplasty in a dialysis patient; immobility and a poor host, but
    #      the kidneys are already gone.
    (
        "59M, ASA 4, for left total knee arthroplasty. Indication: end-stage "
        "osteoarthritis with intractable pain limiting dialysis transfers. PMH: "
        "ESRD on hemodialysis, T2DM, PVD, anemia, secondary "
        "hyperparathyroidism.",
        1, 0, 0, 0,
    ),

    # 438. Bypass surgery in an elderly man with prior stroke; cerebral
    #      vulnerability plus a pump run.
    (
        "77M, ASA 4, for coronary artery bypass grafting x3. Indication: "
        "three-vessel disease with reduced EF 38% and angina. PMH: prior "
        "ischemic CVA with residual dysarthria, carotid disease 60% bilateral, "
        "CKD stage 3, T2DM.",
        0, 0, 1, 1,
    ),

    # 439. Spinal epidural abscess in a diabetic IV drug user; already
    #      bacteremic on arrival.
    (
        "48M, ASA 4E, for emergent T8-T10 laminectomy with evacuation of "
        "epidural abscess. Indication: spinal epidural abscess with progressive "
        "paraparesis, MRSA bacteremia. PMH: IV drug use, T2DM, hepatitis C, "
        "endocarditis. On vancomycin.",
        0, 0, 1, 0,
    ),

    # 440. Aneurysm in a young connective tissue disease patient; elective,
    #      open, but he is 34 with a normal heart and kidneys.
    (
        "34M, ASA 3, for open repair of thoracoabdominal aortic aneurysm, "
        "Crawford IV. Indication: 5.9 cm aneurysm in Loeys-Dietz syndrome. PMH: "
        "Loeys-Dietz syndrome, prior root replacement 2020. Normal creatinine, "
        "never smoker.",
        0, 0, 1, 0,
    ),

    # 441. Paediatric implant; short, healthy child.
    (
        "2F, ASA 2, for left cochlear implantation. Indication: congenital "
        "bilateral profound sensorineural hearing loss, no aided benefit. PMH: "
        "connexin 26 mutation. Otherwise well.",
        0, 0, 0, 0,
    ),

    # 442. Negative appendectomy in a young woman; short, uneventful.
    (
        "24F, ASA 1E, for diagnostic laparoscopy with appendectomy. Indication: "
        "right iliac fossa pain with equivocal imaging, appendix macroscopically "
        "normal at operation. No PMH. Home day 1.",
        0, 0, 0, 0,
    ),

    # 443. Reconstructive urology in an older patient; laparoscopic, short.
    (
        "68F, ASA 3, for laparoscopic pyeloplasty. Indication: symptomatic "
        "pelviureteric junction obstruction with recurrent pyelonephritis, "
        "split function 34%. PMH: HTN, T2DM, BMI 30. Independent.",
        0, 0, 0, 0,
    ),

    # 444. Short obstetric procedure in a young woman.
    (
        "27F, ASA 2E, for hysteroscopic removal of retained products of "
        "conception. Indication: secondary postpartum hemorrhage day 12, "
        "retained placental tissue on ultrasound. PMH: recent vaginal delivery. "
        "Hemoglobin 9.4.",
        0, 0, 0, 0,
    ),

    # 445. Revision shoulder arthroplasty; long case, but a mobile, cognitively
    #      intact patient.
    (
        "72F, ASA 3, for revision right reverse total shoulder arthroplasty. "
        "Indication: glenoid baseplate loosening with pain and pseudoparalysis. "
        "PMH: RA on methotrexate, osteoporosis, HTN. Independent, lives with "
        "spouse.",
        0, 0, 0, 0,
    ),

    # 446. Pericardial drainage in a uremic patient; already on dialysis, so
    #      the AKI label has nowhere to go.
    (
        "63M, ASA 4E, for subxiphoid pericardial window. Indication: uremic "
        "pericardial effusion with tamponade despite intensified dialysis. PMH: "
        "ESRD on hemodialysis, poor compliance with sessions, HTN, anemia.",
        0, 0, 0, 0,
    ),

    # 447. Abdominoperineal resection: long, two-field, permanent stoma, and an
    #      active malignancy.
    (
        "66M, ASA 3, for abdominoperineal resection with end colostomy. "
        "Indication: low rectal adenocarcinoma involving the sphincter complex "
        "after chemoradiation. PMH: HTN, T2DM, BMI 32, former smoker. "
        "Anticipated 5h, prone perineal phase.",
        1, 0, 0, 0,
    ),

    # 448. Awake craniotomy with language mapping; long but she is awake,
    #      young, and cognitively normal.
    (
        "41F, ASA 2, for awake left temporal craniotomy for glioma resection "
        "with language mapping. Indication: low-grade glioma in eloquent "
        "cortex, growth on surveillance. PMH: seizure disorder on "
        "levetiracetam. Neuropsychology baseline normal.",
        0, 0, 0, 0,
    ),

    # 449. Fistula ligation for steal; short, regional, but he remains
    #      dialysis-dependent through a catheter.
    (
        "70M, ASA 4, for ligation of brachiocephalic arteriovenous fistula with "
        "tunnelled catheter insertion. Indication: severe dialysis access steal "
        "syndrome with digital ischemia and rest pain. PMH: ESRD on "
        "hemodialysis, T2DM, PVD, CAD.",
        0, 0, 0, 0,
    ),

    # 450. Invasive fungal sinusitis in a neutropenic patient; the infection is
    #      angioinvasive and already in his lungs.
    (
        "54M, ASA 5E, for emergent endoscopic sinus debridement. Indication: "
        "acute invasive fungal rhinosinusitis with orbital extension in "
        "neutropenia. PMH: AML on induction chemotherapy, neutrophils 0.1, "
        "T2DM, pulmonary infiltrates on CT. On liposomal amphotericin.",
        0, 1, 1, 0,
    ),

    # 451. Radiation enteritis in a patient with a hostile pelvis; long
    #      adhesiolysis and poor tissue.
    (
        "62F, ASA 3, for laparotomy with small bowel resection for radiation "
        "enteritis. Indication: chronic radiation enteritis with subacute "
        "obstruction and weight loss, 9 years after pelvic radiotherapy. PMH: "
        "cervical cancer in remission, prior radiotherapy, albumin 2.9.",
        0, 0, 0, 0,
    ),

    # 452. Inflammatory kidney in a diabetic; the kidney is destroyed and
    #      infected, but the other one is normal.
    (
        "57F, ASA 3, for open left nephrectomy. Indication: xanthogranulomatous "
        "pyelonephritis with staghorn calculus and nonfunctioning kidney, "
        "recurrent sepsis. PMH: T2DM, recurrent UTI, BMI 34. Contralateral "
        "kidney normal.",
        0, 0, 0, 0,
    ),

    # 453. Prophylactic tubal surgery in a healthy woman at the time of another
    #      procedure.
    (
        "44F, ASA 2, for laparoscopic bilateral salpingectomy. Indication: "
        "permanent contraception with opportunistic ovarian cancer risk "
        "reduction. PMH: BMI 27, two prior vaginal deliveries. Day case.",
        0, 0, 0, 0,
    ),

    # 454. Distal femur fracture in an elderly woman with a knee replacement
    #      above it; long recovery, restricted weight-bearing.
    (
        "84F, ASA 4, for distal femoral replacement. Indication: comminuted "
        "periprosthetic distal femur fracture above a total knee arthroplasty "
        "after fall. PMH: TKA 2014, osteoporosis, CHF, CKD stage 3, mild "
        "cognitive impairment.",
        1, 0, 0, 1,
    ),

    # 455. Valve replacement in a middle-aged patient with a good ventricle;
    #      short pump run, fast track.
    (
        "57M, ASA 3, for aortic valve replacement with bioprosthesis. "
        "Indication: severe aortic stenosis with exertional dyspnea, gradient "
        "52 mmHg. PMH: HTN, hyperlipidemia. EF 58%, creatinine 0.9, never "
        "smoker.",
        0, 0, 0, 0,
    ),

    # 456. Short laparoscopic adrenal case in a well patient.
    (
        "48F, ASA 3, for laparoscopic left adrenalectomy. Indication: primary "
        "aldosteronism with lateralisation on adrenal vein sampling, refractory "
        "hypertension. PMH: HTN on four agents, hypokalemia on supplementation.",
        0, 0, 0, 0,
    ),

    # 457. Revision spinal fusion in an obese patient; long prone case,
    #      extensive dissection.
    (
        "58M, ASA 3, for revision L4-S1 fusion with interbody cages. "
        "Indication: pseudarthrosis with recurrent back pain 3 years after "
        "index fusion. PMH: prior lumbar fusion, BMI 40, T2DM, current smoker, "
        "chronic opioid use.",
        1, 0, 0, 0,
    ),

    # 458. Redo varicose vein surgery in a healthy patient; day case.
    (
        "52F, ASA 2, for redo groin exploration with phlebectomies. Indication: "
        "recurrent varicose veins with neovascularisation after prior stripping. "
        "PMH: prior vein surgery 2015, BMI 28. Day case.",
        0, 0, 0, 0,
    ),

    # 459. Airway surgery for sleep apnea in an obese patient; the airway swells
    #      and he is a CO2 retainer.
    (
        "45M, ASA 4, for uvulopalatopharyngoplasty with tonsillectomy. "
        "Indication: severe obstructive sleep apnea, AHI 62, CPAP intolerant. "
        "PMH: BMI 47, obesity hypoventilation, HTN, T2DM, difficult airway "
        "anticipated.",
        0, 1, 0, 0,
    ),

    # 460. Feeding tube after a stroke; short case, but she is aspirating and
    #      already confused.
    (
        "79F, ASA 4, for percutaneous endoscopic gastrostomy. Indication: "
        "persistent dysphagia with failed swallow assessments 3 weeks after "
        "large left MCA infarct. PMH: recent CVA with global aphasia and "
        "hemiparesis, AF, HTN, aspiration pneumonia during admission.",
        0, 1, 0, 1,
    ),

    # 461. Elective gallbladder in an older but well patient; day case.
    (
        "70F, ASA 3, for laparoscopic cholecystectomy. Indication: symptomatic "
        "gallstones with two prior episodes of biliary colic. PMH: HTN, "
        "hypothyroidism, BMI 27. Independent, home the same day.",
        0, 0, 0, 0,
    ),

    # 462. Combined valve and bypass surgery in an elderly patient; long pump
    #      run on marginal kidneys.
    (
        "79M, ASA 4, for aortic valve replacement with coronary artery bypass "
        "grafting x2. Indication: severe aortic stenosis with concomitant "
        "two-vessel disease. PMH: CKD stage 3 Cr 1.7, T2DM, HTN, prior TIA. "
        "Cross-clamp 118 min.",
        0, 1, 1, 1,
    ),

    # 463. Long resection with fluid absorption; the hyponatremia is a
    #      confusional state in an older man.
    (
        "76M, ASA 3, for transurethral resection of prostate. Indication: 110 g "
        "prostate with retention. PMH: HTN, hyperlipidemia. Resection 92 min, "
        "sodium fell to 118 postoperatively, glycine absorption.",
        0, 0, 0, 1,
    ),

    # 464. Fertility-preserving surgery in a young woman; laparoscopic, home
    #      day 1.
    (
        "33F, ASA 2, for laparoscopic myomectomy. Indication: 6 cm "
        "intramural fibroid with recurrent implantation failure, desires "
        "fertility. PMH: subfertility, BMI 26. Two fibroids enucleated.",
        0, 0, 0, 0,
    ),

    # 465. Knee replacement in a rheumatoid patient on biologics; held
    #      preoperatively, uneventful.
    (
        "63F, ASA 3, for left total knee arthroplasty. Indication: end-stage "
        "rheumatoid arthropathy with valgus deformity. PMH: rheumatoid "
        "arthritis on adalimumab, held 2 weeks preoperatively, methotrexate "
        "continued, osteoporosis.",
        0, 0, 0, 0,
    ),

    # 466. Tumour resection in an elderly demented patient; the operation is
    #      short but the brain has no reserve.
    (
        "82F, ASA 4, for right frontal craniotomy for meningioma resection. "
        "Indication: 5 cm meningioma with vasogenic edema and worsening "
        "confusion. PMH: moderate dementia, HTN, AF on apixaban, CKD stage 3. "
        "On dexamethasone.",
        0, 0, 0, 1,
    ),

    # 467. Carotid surgery in an elderly but well man; short case, awake under
    #      regional block.
    (
        "82M, ASA 3, for left carotid endarterectomy. Indication: symptomatic "
        "80% stenosis after amaurosis fugax. PMH: HTN, hyperlipidemia, former "
        "smoker. Independent, cognitively sharp, cervical plexus block, awake "
        "throughout.",
        0, 0, 0, 0,
    ),

    # 468. Tracheostomy in a superobese patient; short neck, difficult access,
    #      and a chest that does not clear secretions.
    (
        "48M, ASA 4, for open tracheostomy. Indication: failed extubation x2 "
        "after prolonged ventilation for pneumonia. PMH: BMI 58, obesity "
        "hypoventilation, T2DM, OSA, immobile. Day 14 of ventilation.",
        0, 1, 0, 1,
    ),

    # 469. Elective body contouring in a healthy patient; long but light.
    (
        "40F, ASA 2, for abdominoplasty with rectus plication. Indication: "
        "abdominal wall laxity and diastasis after two pregnancies, elective. "
        "PMH: BMI 26, nonsmoker. Mechanical prophylaxis, mobilised same day.",
        0, 0, 0, 0,
    ),

    # 470. Empyema in a child; decorticated, chest tube, home in a week.
    (
        "6M, ASA 2, for VATS decortication. Indication: parapneumonic empyema "
        "with loculations after community-acquired pneumonia, failed drainage. "
        "PMH: none. Febrile, on IV antibiotics.",
        0, 0, 0, 0,
    ),

    # 471. Appendicitis in an immunosuppressed patient; blunted signs, late
    #      presentation, but the operation is short.
    (
        "45F, ASA 3E, for laparoscopic appendectomy. Indication: acute "
        "appendicitis with delayed diagnosis on background immunosuppression, "
        "3 days of symptoms. PMH: renal transplant 2018 on tacrolimus and "
        "prednisone, baseline Cr 1.4.",
        0, 0, 1, 0,
    ),

    # 472. Pancreas transplant alone after a failed prior graft; heavily
    #      immunosuppressed with enteric drainage.
    (
        "45M, ASA 4, for pancreas transplant alone. Indication: type 1 diabetes "
        "with severe hypoglycemia unawareness, functioning kidney transplant. "
        "PMH: T1DM 30 years, kidney transplant 2020, prior failed pancreas "
        "graft 2022, gastroparesis.",
        1, 1, 1, 0,
    ),

    # 473. Blunt liver trauma in an older patient; packed, transfused, and the
    #      physiology does not tolerate it the way a 30-year-old's would.
    (
        "71M, ASA 5E, for emergent laparotomy with hepatic packing and "
        "damage control closure. Indication: grade V liver injury after motor "
        "vehicle collision, in extremis. PMH: HTN, CAD, CKD stage 3. 16 units "
        "transfused, coagulopathic.",
        1, 1, 1, 1,
    ),

    # 474. Prophylactic colectomy in a young FAP patient; laparoscopic, home in
    #      four days.
    (
        "22M, ASA 1, for laparoscopic total proctocolectomy with ileal "
        "pouch-anal anastomosis and diverting ileostomy. Indication: familial "
        "adenomatous polyposis with hundreds of polyps. PMH: FAP, APC "
        "pathogenic variant. Otherwise well.",
        0, 0, 0, 0,
    ),

    # 475. Urethral reconstruction in an older man; long lithotomy but a well
    #      patient.
    (
        "67M, ASA 3, for perineal urethrostomy. Indication: long panurethral "
        "stricture after radiotherapy, recurrent retention, unsuitable for "
        "graft. PMH: prostate cancer s/p radiotherapy 2019, HTN, T2DM.",
        0, 0, 0, 0,
    ),

    # 476. Hysterectomy in a woman with a scarred pelvis; longer than planned
    #      but she is 43 and well.
    (
        "43F, ASA 2, for total abdominal hysterectomy. Indication: chronic "
        "pelvic pain with dense adhesions after three prior cesareans, failed "
        "conservative management. PMH: three cesareans, BMI 30. Bladder "
        "adhesiolysis required.",
        0, 0, 0, 0,
    ),

    # 477. Ankle fracture in a diabetic with neuropathy; poor tissue and slow
    #      healing, but no systemic insult.
    (
        "58M, ASA 3, for open reduction internal fixation of right ankle. "
        "Indication: trimalleolar fracture-dislocation after fall. PMH: T2DM "
        "HbA1c 8.9, peripheral neuropathy, obesity BMI 37, HTN. "
        "Non-weight-bearing 8 weeks.",
        0, 0, 0, 0,
    ),

    # 478. Mitral surgery in an elderly woman with a dilated atrium; long pump
    #      run in a frail patient.
    (
        "83F, ASA 4, for mitral valve repair with annuloplasty and left atrial "
        "appendage occlusion. Indication: severe degenerative mitral "
        "regurgitation with heart failure. PMH: permanent AF, CKD stage 3, "
        "frailty, prior falls, HTN.",
        0, 0, 1, 1,
    ),

    # 479. Brain metastasis in a smoker with a burdened chest; the head does
    #      fine but the lungs do not.
    (
        "65M, ASA 4, for left cerebellar craniotomy for resection of solitary "
        "metastasis. Indication: 3 cm cerebellar metastasis from NSCLC with "
        "obstructive hydrocephalus. PMH: NSCLC, COPD FEV1 44%, current smoker "
        "45 pack-years, on dexamethasone.",
        0, 1, 0, 0,
    ),

    # 480. DISCORDANT vs rows 9, 44 and 82: elective open aneurysm repair, but
    #      infrarenal clamp, fit patient, normal kidneys, epidural.
    (
        "68M, ASA 3, for elective open repair of infrarenal abdominal aortic "
        "aneurysm. Indication: 5.7 cm aneurysm, unsuitable anatomy for EVAR. "
        "PMH: HTN, former smoker quit 12 years ago. Creatinine 0.9, normal PFTs, "
        "epidural.",
        0, 0, 0, 0,
    ),

    # 481. Locally advanced thyroid cancer with airway invasion; the trachea is
    #      resected and the swallow is disrupted.
    (
        "68F, ASA 4, for total thyroidectomy with tracheal window resection and "
        "reconstruction. Indication: locally advanced papillary carcinoma with "
        "tracheal invasion and hemoptysis. PMH: HTN, T2DM, prior neck "
        "radiotherapy.",
        0, 1, 0, 0,
    ),

    # 482. Liver resection in a cirrhotic; the parenchyma is the problem and
    #      the remnant is marginal.
    (
        "64M, ASA 4, for right posterior sectionectomy. Indication: "
        "hepatocellular carcinoma 5 cm in hepatitis C cirrhosis, Child-Pugh A, "
        "portal pressure acceptable. PMH: hepatitis C cirrhosis, "
        "thrombocytopenia, esophageal varices, T2DM.",
        0, 0, 1, 0,
    ),

    # 483. Bladder tumour resection in a young patient; short, spinal, day
    #      case.
    (
        "44M, ASA 2, for transurethral resection of bladder tumour. Indication: "
        "painless visible hematuria with a 2 cm papillary bladder lesion on "
        "cystoscopy. PMH: former smoker 10 pack-years, otherwise well.",
        0, 0, 0, 0,
    ),

    # 484. Ovarian malignancy in a young woman; fertility-sparing, but the
    #      cancer still counts.
    (
        "29F, ASA 2, for laparoscopic right salpingo-oophorectomy with staging "
        "biopsies. Indication: stage IA mucinous ovarian carcinoma, "
        "fertility-sparing intent. PMH: none. Contralateral ovary and uterus "
        "preserved.",
        1, 0, 0, 0,
    ),

    # 485. High-energy hip fracture in a young adult; fixed urgently, mobilised
    #      the next day.
    (
        "32M, ASA 2E, for open reduction internal fixation of left femoral neck "
        "with cannulated screws. Indication: displaced femoral neck fracture "
        "after fall from height at work. PMH: none. To OR within 6h.",
        0, 0, 0, 0,
    ),

    # 486. Emergency bypass after a failed intervention; cardiogenic shock,
    #      balloon pump, and a stunned circulation.
    (
        "64M, ASA 5E, for emergent coronary artery bypass grafting x3. "
        "Indication: left main dissection during PCI with cardiogenic shock, on "
        "intra-aortic balloon pump. PMH: CAD, T2DM, HTN, CKD stage 2. Lactate "
        "5.4, on two inotropes.",
        0, 1, 1, 1,
    ),

    # 487. Lumbar decompression in an obese patient; prone, longer than usual,
    #      but no systemic insult.
    (
        "55M, ASA 3, for L3-L5 decompression. Indication: lumbar canal stenosis "
        "with neurogenic claudication limiting walking to 50 m. PMH: BMI 46, "
        "OSA on CPAP, T2DM, HTN. Mobilised day 1, home day 2.",
        0, 0, 0, 0,
    ),

    # 488. Amputation in an elderly demented woman; immobile, confused at
    #      baseline, and the ward finishes what the disease started.
    (
        "86F, ASA 4, for left below-knee amputation. Indication: dry gangrene of "
        "the forefoot with unreconstructable disease and rest pain. PMH: "
        "moderate dementia, PVD, T2DM, CKD stage 3, AF. Nursing home, hoist "
        "transfers.",
        1, 0, 0, 1,
    ),

    # 489. Neck dissection in a fit patient; long but no radiotherapy history
    #      and a normal swallow.
    (
        "51M, ASA 2, for selective neck dissection levels I-III with transoral "
        "resection of tongue lesion. Indication: T1N1 oral tongue squamous cell "
        "carcinoma, HPV negative. PMH: former smoker quit 8 years ago. No prior "
        "radiotherapy.",
        0, 0, 0, 0,
    ),

    # 490. Gallbladder drainage in a patient too sick for cholecystectomy; the
    #      sepsis is what drives everything.
    (
        "88M, ASA 5E, for open cholecystostomy. Indication: acute cholecystitis "
        "with septic shock, unfit for cholecystectomy, percutaneous access "
        "failed. PMH: CHF EF 25%, CKD stage 4, dementia, AF. On norepinephrine.",
        0, 1, 1, 1,
    ),

    # 491. Prostatectomy in an obese patient; longer robotic case, but he is 55
    #      and well.
    (
        "55M, ASA 3, for robotic radical prostatectomy. Indication: prostate "
        "adenocarcinoma Gleason 3+4, PSA 9.2, clinically localised. PMH: BMI 42, "
        "T2DM, OSA on CPAP, HTN. Steep Trendelenburg, 4h case.",
        0, 0, 0, 0,
    ),

    # 492. Placenta previa with bleeding; transfused, but she is 30 and
    #      recovers immediately.
    (
        "30F, ASA 3E, for emergent cesarean section at 34 weeks. Indication: "
        "major placenta previa with antepartum hemorrhage, 1.2 L blood loss. "
        "PMH: prior cesarean, placenta previa on surveillance. Three units "
        "transfused, uterus preserved.",
        0, 0, 0, 0,
    ),

    # 493. Scoliosis correction in a healthy adolescent; long prone case with
    #      blood loss, but a pristine host.
    (
        "15F, ASA 1, for T4-L3 posterior spinal instrumented fusion. "
        "Indication: adolescent idiopathic scoliosis, Cobb angle 62 degrees, "
        "progressive. No PMH. Cell salvage used, EBL 900 mL.",
        0, 0, 0, 0,
    ),

    # 494. Bypass surgery in a woman with diabetes and a small target vessel;
    #      routine pump run, uneventful.
    (
        "68F, ASA 3, for coronary artery bypass grafting x3. Indication: "
        "three-vessel disease with diabetes, preferred over PCI. PMH: T2DM on "
        "insulin, HTN, hyperlipidemia. EF 55%, creatinine 0.8, never smoker.",
        0, 0, 0, 0,
    ),

    # 495. Craniotomy in a patient on active chemotherapy; neutropenic-adjacent
    #      and deconditioned, but the head recovers.
    (
        "49F, ASA 4, for right frontal craniotomy for resection of brain "
        "metastasis. Indication: solitary 3 cm metastasis from triple-negative "
        "breast cancer with edema. PMH: metastatic breast cancer on "
        "chemotherapy, port in situ, neutrophils 1.4, prior DVT.",
        1, 0, 0, 0,
    ),

    # 496. Distal bypass in an elderly diabetic; regional, limb salvaged, home
    #      in a week.
    (
        "77M, ASA 4, for left femoral-below-knee popliteal bypass with "
        "prosthetic graft. Indication: critical limb ischemia with a "
        "nonhealing toe ulcer, no suitable vein. PMH: T2DM, PVD, CAD, CKD stage "
        "3, HTN. Epidural anesthesia.",
        0, 0, 0, 0,
    ),

    # 497. Adult tonsillectomy for sleep apnea in an obese patient; the airway
    #      swells but he is 36 and manages.
    (
        "36M, ASA 3, for tonsillectomy. Indication: obstructive sleep apnea with "
        "grade 4 tonsillar hypertrophy, AHI 34, CPAP intolerant. PMH: BMI 36, "
        "HTN. Overnight observation with pulse oximetry.",
        0, 0, 0, 0,
    ),

    # 498. Stoma closure in an older patient; short, and she is well.
    (
        "76F, ASA 3, for loop ileostomy closure. Indication: restoration of "
        "continuity 6 months after anterior resection, anastomosis intact on "
        "contrast study. PMH: rectal cancer in remission, HTN, hypothyroidism. "
        "Independent.",
        0, 0, 0, 0,
    ),

    # 499. Knee replacement in a young patient after trauma; fit, mobile,
    #      motivated.
    (
        "48M, ASA 2, for right total knee arthroplasty. Indication: "
        "post-traumatic arthritis 15 years after tibial plateau fracture. PMH: "
        "prior tibial plateau ORIF. Nonsmoker, BMI 27, mobilised day 0, home day "
        "1.",
        0, 0, 0, 0,
    ),

    # 500. Last row, and a deliberately quiet one: elective, healthy, brief.
    (
        "36F, ASA 1, for laparoscopic appendectomy. Indication: acute "
        "appendicitis, 14h of symptoms, no perforation on ultrasound. No PMH. "
        "Home day 1.",
        0, 0, 0, 0,
    ),

]

_EXPECTED_N = 500
assert len(_CURATED_ROWS) == _EXPECTED_N, (
    f"Curated dataset is {len(_CURATED_ROWS)} rows, expected {_EXPECTED_N}. "
    "Rows are hand-written; a mismatch means one was lost or duplicated in an edit."
)


def get_pseudo_data():
    """
    Return the hand-curated dataset of preoperative notes and outcomes.

    Takes no arguments and always returns all 500 curated rows. There is no
    subsetting parameter: the rows are a designed set, not a sample, and the
    prevalence and discordance structure only holds for the whole thing.

    Returns
    -------
    pandas.DataFrame with columns:
        clinical_note (str), DVT (int), Pneumonia (int), AKI (int),
        Delirium (int)
    """
    return pd.DataFrame(
        [
            {
                "clinical_note": note,
                "DVT": dvt,
                "Pneumonia": pna,
                "AKI": aki,
                "Delirium": delirium,
            }
            for note, dvt, pna, aki, delirium in _CURATED_ROWS
        ]
    )
