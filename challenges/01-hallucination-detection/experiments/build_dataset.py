"""Build the Challenge 01 benchmark dataset (100 examples).

Design:
- 50 hand-written base items set in an Indian context and written in Indian
  English (Indian names and places, rupees, lakh and crore, "carried forward").
  Each has a fictional context, a question and 2 faithful answer claims written
  as paraphrases (not copies) of the context.
- Each base item also has one hand-written perturbation that makes the answer
  unsupported. The perturbation is tagged with a hallucination type.
- Every base item yields two examples: one faithful, one hallucinated -> 100.
- Contexts describe fictional entities so a model cannot answer from memory;
  "supported" strictly means "supported by the given context".
- Split by base item (odd -> dev, even -> test) so a faithful/hallucinated
  pair never straddles splits. Thresholds are tuned on dev, reported on test.

Run:  python challenges/01-hallucination-detection/experiments/build_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "datasets" / "hallucination_qa" / "v2.jsonl"

# (domain, context, question, [faithful claims], (type, index | None for append, text))
BASE = [
    ("company", "Sahyadri Robotics was founded in 2014 by Meera Kulkarni in Pune. The company makes warehouse picking robots for e-commerce firms. In 2022 it had 340 employees and opened a second office in Bengaluru.",
     "Who founded Sahyadri Robotics and where?",
     ["Sahyadri Robotics was started by Meera Kulkarni.", "The company was set up in Pune in 2014."],
     ("entity_swap", 0, "Sahyadri Robotics was started by Rohan Deshpande.")),
    ("product", "The Pavan X2 air purifier has a HEPA-13 filter that should be replaced every 9 months. It covers rooms up to 45 square metres. The filter indicator light turns red when replacement is due.",
     "How often should the Pavan X2 filter be replaced?",
     ["The Pavan X2 filter should be changed every 9 months.", "A red indicator light shows when the filter is due."],
     ("number_change", 0, "The Pavan X2 filter should be changed every 6 months.")),
    ("hr_policy", "Employees at Ganga Logistics earn 1.5 days of paid leave per month. Up to 10 days of unused leave can be carried forward to the next year. Leave applications must be submitted at least two weeks in advance.",
     "How much unused leave can Ganga Logistics employees carry forward?",
     ["Employees can carry forward up to 10 unused leave days to the next year.", "Leave is earned at 1.5 days per month."],
     ("unsupported_addition", None, "Any carried-forward leave lapses if not used by 31 March.")),
    ("medical", "Zelvanor is approved for treating moderate plaque psoriasis in adults. The recommended dose is 40 mg injected every two weeks. Common side effects include redness at the injection site and headache. It is not approved for patients below 18 years of age.",
     "Can Zelvanor be given to children?",
     ["No, Zelvanor is not approved for patients younger than 18.", "It is approved for adults with moderate plaque psoriasis."],
     ("negation", 0, "Yes, Zelvanor is approved for patients younger than 18.")),
    ("history", "The town of Rampur Kalan received its municipal charter in 1883. Its first chairman, Lala Hardayal Gupta, oversaw the building of the stone bridge over the Sukhi river in 1890. The bridge was rebuilt after the flood of 1956.",
     "When was the stone bridge in Rampur Kalan built?",
     ["The stone bridge over the Sukhi river was built in 1890.", "Its construction was overseen by Lala Hardayal Gupta, the town's first chairman."],
     ("date_change", 0, "The stone bridge over the Sukhi river was built in 1883.")),
    ("software", "Version 3.2 of the Kaveri SDK deprecates the sync_upload method. Developers should migrate to upload_async, which supports retries. The sync_upload method will be removed in version 4.0.",
     "What should developers use instead of sync_upload?",
     ["Developers should switch to upload_async.", "The upload_async method supports retries."],
     ("unsupported_addition", None, "The upload_async method is also twice as fast as sync_upload.")),
    ("science", "The Kesar glacier in Uttarakhand has retreated 2.3 kilometres since 1950, according to a 2021 survey. The survey team was led by the Himalayan Ice Institute, Dehradun. Meltwater from the glacier feeds the Kesar river.",
     "How far has the Kesar glacier retreated?",
     ["The Kesar glacier has pulled back 2.3 kilometres since 1950.", "This figure comes from a 2021 survey."],
     ("number_change", 0, "The Kesar glacier has pulled back 3.2 kilometres since 1950.")),
    ("hospitality", "Annapurna Bhojanalaya is open from Tuesday to Sunday, 12 pm to 11 pm. It remains closed on Mondays. Table booking is compulsory for groups of six or more.",
     "Is Annapurna Bhojanalaya open on Mondays?",
     ["No, Annapurna Bhojanalaya is closed on Mondays.", "It is open Tuesday to Sunday, 12 pm to 11 pm."],
     ("negation", 0, "Yes, Annapurna Bhojanalaya is open on Mondays.")),
    ("finance", "Shakti Cooperative Bank's savings account pays 3.1% annual interest on balances above Rs 5 lakh. Balances below Rs 5 lakh earn 2.7%. There is no monthly maintenance charge.",
     "What interest does Shakti Cooperative Bank pay on balances above Rs 5 lakh?",
     ["Balances above Rs 5 lakh earn 3.1% a year.", "The account has no monthly maintenance charge."],
     ("in_context_swap", 0, "Balances above Rs 5 lakh earn 2.7% a year.")),
    ("sports", "The Kochi Blasters won the 2019 state football league, defeating Thrissur United 3-1 in the final. Striker Anjali Nair scored two of the goals. The team's coach was Joseph Mathew.",
     "Who scored two goals in the 2019 final?",
     ["Anjali Nair scored twice in the final.", "The Blasters beat Thrissur United 3-1."],
     ("in_context_swap", 0, "Joseph Mathew scored twice in the final.")),
    ("travel", "The Deccan Shuttle bus runs between Swargate and Hinjewadi every 30 minutes on weekdays. On weekends it runs every hour. A single ticket costs Rs 45.",
     "How often does the Deccan Shuttle run on weekends?",
     ["On weekends the Deccan Shuttle runs once an hour.", "On weekdays buses come every 30 minutes."],
     ("number_change", 1, "On weekdays buses come every 15 minutes.")),
    ("education", "Narmada University requires a minimum CGPA of 7.0 for admission to its M.Tech in data science. Applicants must also submit two letters of recommendation. The programme lasts 18 months.",
     "How long is Narmada University's M.Tech in data science?",
     ["The programme takes 18 months to complete.", "Applicants need two recommendation letters."],
     ("unsupported_addition", None, "It can also be completed fully online.")),
    ("hardware", "The Tejas-7 microcontroller runs at 133 MHz and has 264 KB of RAM. It supports USB 1.1 and has 26 GPIO pins.",
     "How much RAM does the Tejas-7 have?",
     ["The Tejas-7 has 264 KB of RAM.", "It runs at a clock speed of 133 MHz."],
     ("number_change", 0, "The Tejas-7 has 512 KB of RAM.")),
    ("legal", "Under the Navi Nagar rent control rules, landlords must give tenants 60 days' written notice before increasing the rent. Increases are capped at 5% per year. The rules apply to buildings with four or more flats.",
     "How much notice must Navi Nagar landlords give before increasing rent?",
     ["Landlords must give 60 days of written notice.", "Annual increases are limited to 5%."],
     ("overgeneralization", 1, "Annual increases are limited to 5% for every rented property in Navi Nagar.")),
    ("device", "The DhadkanTrack band measures heart rate every 5 seconds and stores 14 days of data. It is water resistant up to 50 metres. It is not a medical device and should not be used to diagnose any condition.",
     "Can DhadkanTrack be used to diagnose heart conditions?",
     ["No, DhadkanTrack should not be used for diagnosis.", "It is not classified as a medical device."],
     ("negation", 1, "It is a certified medical device.")),
    ("company", "Swadesh Foods acquired Hariyali Farms in March 2020 for Rs 48 crore. Hariyali's founder, Lalita Reddy, joined Swadesh as head of sourcing.",
     "How much did Swadesh Foods pay for Hariyali Farms?",
     ["Swadesh Foods paid Rs 48 crore for Hariyali Farms.", "The deal was closed in March 2020."],
     ("date_change", 1, "The deal was closed in March 2021.")),
    ("space", "The probe Tara was launched in 2031 and reached the asteroid Kalpana in 2034. It collected 120 grams of surface material. The sample capsule is scheduled to return to Earth in 2037.",
     "How much material did Tara collect?",
     ["Tara gathered 120 grams of material from the surface of Kalpana.", "It reached Kalpana in 2034."],
     ("in_context_swap", 1, "It reached Kalpana in 2031.")),
    ("support", "Orders from Desi Threads are dispatched within 2 working days. Delivery is free on orders above Rs 999. Returns are accepted within 30 days if items are unused and have their tags intact.",
     "What is the return window at Desi Threads?",
     ["Desi Threads accepts returns within 30 days.", "Items must be unused to be returned."],
     ("unsupported_addition", None, "Return pickup is always free.")),
    ("agriculture", "The Sona-68 tomato variety matures in 68 days and is resistant to fusarium wilt. It grows best in full sun and well-drained soil.",
     "How long does the Sona-68 tomato take to mature?",
     ["Sona-68 tomatoes mature in about 68 days.", "The variety resists fusarium wilt."],
     ("entity_swap", 1, "The variety resists late blight.")),
    ("hr_policy", "Tarang Labs offers 26 weeks of paid maternity leave to all employees after six months of service. Employees can also request a phased return of up to four weeks.",
     "How much paid maternity leave does Tarang Labs offer?",
     ["Tarang Labs gives 26 weeks of paid maternity leave.", "Employees qualify after six months of service."],
     ("overgeneralization", 1, "Every employee qualifies from their first day.")),
    ("energy", "The Thar Wind Park has 42 turbines and a capacity of 150 megawatts. It began operating in 2018 and supplies power to about 60,000 homes.",
     "How many homes does the Thar Wind Park power?",
     ["It supplies electricity to roughly 60,000 homes.", "The park has a capacity of 150 megawatts."],
     ("number_change", 0, "It supplies electricity to roughly 90,000 homes.")),
    ("museum", "The Vidisha Museum's east wing houses its collection of Gupta-era coins. The west wing is dedicated to modern sculpture. Entry is free on the first Sunday of every month.",
     "What is in the Vidisha Museum's east wing?",
     ["The east wing holds the museum's Gupta-era coin collection.", "Entry is free on the first Sunday of each month."],
     ("in_context_swap", 0, "The east wing holds the museum's modern sculpture collection.")),
    ("medical", "Doravex should be taken after meals to reduce stomach upset. Patients should avoid grapefruit juice while taking it, because it can raise drug levels.",
     "Should Doravex be taken after food?",
     ["Yes, Doravex should be taken after meals.", "Grapefruit juice should be avoided because it can raise drug levels."],
     ("unsupported_addition", None, "Alcohol should also be avoided for 48 hours after each dose.")),
    ("climate", "Average rainfall in the Kodanad valley is 2,640 mm per year, most of it falling between June and September during the south-west monsoon. Winters are dry, with less than 20 mm of rain per month.",
     "When does most rain fall in the Kodanad valley?",
     ["Most rain falls from June to September.", "The valley gets about 2,640 mm of rain a year."],
     ("entity_swap", 0, "Most rain falls from December to February.")),
    ("automotive", "The Vayu EV has a 40 kWh battery and a certified range of 480 km. It can charge from 10% to 80% in 56 minutes on a 50 kW fast charger.",
     "What is the range of the Vayu EV?",
     ["The Vayu EV has a certified range of 480 km.", "It has a 40 kWh battery."],
     ("number_change", 0, "The Vayu EV has a certified range of 520 km.")),
    ("api_docs", "The Pragati API rate limit is 100 requests per minute per key. Exceeding the limit returns HTTP 429. Enterprise plans can request a limit of up to 1,000 requests per minute.",
     "What happens when you exceed the Pragati API rate limit?",
     ["Going over the limit returns an HTTP 429 error.", "The default limit is 100 requests per minute per key."],
     ("unsupported_addition", None, "The key is suspended for 24 hours after three violations.")),
    ("biography", "Painter Kamala Iyer was born in Madurai in 1921 and moved to Mumbai in 1948. Her best-known work, Harbour at Dusk, is held by the Sir Ratan Gallery.",
     "Where is Harbour at Dusk held?",
     ["Harbour at Dusk is in the collection of the Sir Ratan Gallery.", "Kamala Iyer moved to Mumbai in 1948."],
     ("entity_swap", 0, "Harbour at Dusk is in the collection of the National Gallery of Modern Art.")),
    ("security", "At Navya Infotech, passwords must be at least 14 characters long and are changed every 180 days. Multi-factor authentication is mandatory for remote access.",
     "How long must passwords be at Navya Infotech?",
     ["Passwords at Navya Infotech must have at least 14 characters.", "MFA is mandatory for remote access."],
     ("overgeneralization", 1, "MFA is mandatory for every login, including from office.")),
    ("urban", "The Gulmohar Park redevelopment will add 3 hectares of green space and a 400-metre jogging track. Work starts in April and is expected to take 14 months.",
     "How long will the Gulmohar Park work take?",
     ["The work is expected to take 14 months.", "It will begin in April."],
     ("date_change", 1, "It will begin in September.")),
    ("nutrition", "A 100 g serving of Brand K millet bars contains 12 g of protein and 8 g of sugar. The bars are gluten-free but are made in a facility that also processes nuts.",
     "Are Brand K millet bars safe for people with nut allergies?",
     ["They are made in a facility that also processes nuts.", "The bars are gluten-free."],
     ("negation", 0, "They are made in a nut-free facility.")),
    ("telecom", "Jhatpat Mobile's Basic plan includes 2 GB of data per day for Rs 299 a month. Unused daily data does not carry forward. Calls within India are unlimited.",
     "Does unused data carry forward on the Jhatpat Basic plan?",
     ["No, unused data does not carry forward.", "The Basic plan includes 2 GB per day for Rs 299 a month."],
     ("negation", 0, "Yes, unused data carries forward to the next day.")),
    ("archaeology", "Excavations at Tilwara Khera in 2016 uncovered a granary dating to around 1200 BCE. The dig was directed by Dr. Hema Joshi of Aravalli University.",
     "Who directed the Tilwara Khera excavation?",
     ["The excavation was led by Dr. Hema Joshi.", "She is from Aravalli University."],
     ("entity_swap", 1, "She is from Deccan College, Pune.")),
    ("manufacturing", "Line 4 at the Hosur plant produces 1,200 units per shift. Each shift lasts 8 hours, and the line runs two shifts per day. Planned maintenance happens every Sunday.",
     "How many units does Line 4 produce per day?",
     ["Line 4 makes 1,200 units per shift.", "It runs two shifts a day, so about 2,400 units daily."],
     ("number_change", 1, "It runs three shifts a day, so about 3,600 units daily.")),
    ("software", "Lekhak Editor 5.0 adds real-time collaboration and a dark theme. Plugins written for version 4 must be updated to the new API before they will load.",
     "Do version 4 plugins work in Lekhak Editor 5.0?",
     ["No, version 4 plugins need to be updated to the new API first.", "Version 5.0 also adds a dark theme."],
     ("unsupported_addition", None, "An automatic migration tool converts old plugins.")),
    ("zoology", "The Doddabetta shrew lives at elevations between 1,800 and 2,600 metres. It feeds mainly on roots and bulbs and is active year-round.",
     "Does the Doddabetta shrew hibernate?",
     ["No, it stays active all year.", "It eats mostly roots and bulbs."],
     ("negation", 0, "Yes, it hibernates through the winter.")),
    ("insurance", "The Silver health plan covers 70% of in-network hospital costs after a Rs 15,000 deductible. Out-of-network treatment is not covered except in emergencies.",
     "Does the Silver health plan cover out-of-network treatment?",
     ["Only in emergencies; otherwise out-of-network treatment is not covered.", "In-network hospital costs are covered at 70% after the deductible."],
     ("overgeneralization", 0, "Yes, out-of-network treatment is covered at the same 70% rate.")),
    ("gaming", "Rangbhoomi Online's Diwali update adds a new region called Swarnagiri and raises the level cap from 60 to 70. The update releases on 12 November.",
     "What is the new level cap in Rangbhoomi Online?",
     ["The level cap goes up to 70.", "The update also adds the Swarnagiri region."],
     ("in_context_swap", 0, "The level cap goes up to 60.")),
    ("utility", "Residents of Kanakpura Layout may water gardens only on Tuesdays and Saturdays during the summer restriction period. Fines start at Rs 1,000 for a first violation.",
     "What is the fine for a first watering violation in Kanakpura Layout?",
     ["A first violation attracts a fine starting at Rs 1,000.", "Garden watering is allowed on Tuesdays and Saturdays."],
     ("entity_swap", 1, "Garden watering is allowed on Mondays and Thursdays.")),
    ("research", "The study, run at three hospitals in Chennai, enrolled 412 participants and found that the intervention reduced average sleep onset time by 11 minutes. The authors note that the effect was not significant in participants over 65.",
     "Did the intervention work for people over 65?",
     ["The effect was not significant for participants older than 65.", "Overall, sleep onset time dropped by 11 minutes on average."],
     ("overgeneralization", 0, "It worked equally well across all age groups.")),
    ("logistics", "Containers arriving at Port Saurashtra clear customs in an average of 36 hours. Reefer containers are prioritised and typically clear within 12 hours.",
     "How quickly do reefer containers clear customs at Port Saurashtra?",
     ["Reefer containers usually clear within 12 hours.", "They are given priority."],
     ("in_context_swap", 0, "Reefer containers usually clear within 36 hours.")),
    ("library", "The Saraswati Public Library lends up to 15 books per card for three weeks. Books can be renewed twice unless another member has placed a hold.",
     "How many times can books be renewed?",
     ["Books can be renewed two times.", "Renewal is not possible if someone else has a hold on the book."],
     ("number_change", 0, "Books can be renewed five times.")),
    ("cloud", "Megh Storage replicates data across three availability zones in a region. Cross-region replication is available as an add-on for Rs 1.5 per GB.",
     "Does Megh Storage replicate data across regions by default?",
     ["No, cross-region replication is a paid add-on.", "By default data is copied across three availability zones within a region."],
     ("unsupported_addition", None, "Cross-region replication completes within 15 minutes.")),
    ("events", "The Malabar Music Festival runs from 18 to 21 January at Fort Kochi beach. A single-day pass costs Rs 1,500, and a full pass costs Rs 5,000.",
     "How much is a full festival pass?",
     ["A full pass costs Rs 5,000.", "The festival is held at Fort Kochi beach."],
     ("entity_swap", 1, "The festival is held at Marine Drive.")),
    ("botany", "The Pahadi ghost orchid flowers only once a year, usually in July, and each bloom lasts about two weeks. It has no leaves and photosynthesises through its roots.",
     "How does the Pahadi ghost orchid photosynthesise?",
     ["It photosynthesises through its roots, since it has no leaves.", "Each bloom lasts around two weeks."],
     ("unsupported_addition", None, "It is pollinated only by the giant hawk moth.")),
    ("payments", "Refunds through ChaloPay are credited to the original payment method within 5 to 7 working days. Refunds for purchases older than 90 days need manual review.",
     "How long do ChaloPay refunds take?",
     ["Refunds arrive in 5 to 7 working days.", "They go back to the original payment method."],
     ("number_change", 0, "Refunds arrive in 1 to 2 working days.")),
    ("government", "The Sitapur municipal council has 11 members elected for five-year terms. By-elections are held within six months of a seat falling vacant.",
     "How long is a Sitapur council term?",
     ["Council members serve five-year terms.", "By-elections are held within six months of a vacancy."],
     ("in_context_swap", 0, "Council members serve six-month terms.")),
    ("retail", "The Lakeview Mall store's festive hours are 10 am to 10 pm from 1 to 23 December. The store closes at 4 pm on 24 December and is closed on 25 December.",
     "Is the Lakeview Mall store open on 25 December?",
     ["No, the store is closed on 25 December.", "On 24 December it closes at 4 pm."],
     ("negation", 0, "Yes, the store is open from 10 am to 10 pm on 25 December.")),
    ("chemistry", "Compound TX-9 is stable at room temperature but decomposes above 180 C. It is soluble in ethanol and insoluble in water.",
     "Is TX-9 soluble in water?",
     ["No, TX-9 does not dissolve in water.", "It is soluble in ethanol."],
     ("entity_swap", 1, "It is soluble in acetone.")),
    ("transport", "Bicycles are allowed on Metro Line 3 trains outside peak hours, which are 8 to 11 am and 5 to 8 pm on weekdays. Folding cycles are allowed at all times.",
     "Can I take a regular cycle on Metro Line 3 at 9 am on a weekday?",
     ["No, 9 am on a weekday is within peak hours, when regular cycles are not allowed.", "Folding cycles are permitted at any time."],
     ("overgeneralization", 1, "All cycles are permitted at any time.")),
    ("ai", "The Vani-2 language model was trained on 1.4 trillion tokens, including text in 12 Indian languages, and has a context window of 32,000 tokens. It was released under the Apache 2.0 licence in May 2025.",
     "Which licence is Vani-2 released under?",
     ["Vani-2 is released under the Apache 2.0 licence.", "It has a 32,000-token context window."],
     ("unsupported_addition", None, "Commercial use requires a separate agreement.")),
]


def build() -> list[dict]:
    rows: list[dict] = []
    for i, (domain, context, question, claims, (htype, idx, text)) in enumerate(BASE, start=1):
        split = "dev" if i % 2 else "test"
        base_id = f"b{i:03d}"
        faithful = [{"text": c, "supported": True} for c in claims]
        rows.append({
            "id": f"{base_id}-f", "base_id": base_id, "split": split, "domain": domain,
            "question": question, "context": context,
            "response": " ".join(c["text"] for c in faithful),
            "claims": faithful, "hallucinated": False, "hallucination_type": None,
        })
        halluc = [dict(c) for c in faithful]
        if idx is None:
            halluc.append({"text": text, "supported": False})
        else:
            halluc[idx] = {"text": text, "supported": False}
        rows.append({
            "id": f"{base_id}-h", "base_id": base_id, "split": split, "domain": domain,
            "question": question, "context": context,
            "response": " ".join(c["text"] for c in halluc),
            "claims": halluc, "hallucinated": True, "hallucination_type": htype,
        })
    return rows


def main() -> None:
    rows = build()
    assert len(rows) == 100, len(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} examples to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
