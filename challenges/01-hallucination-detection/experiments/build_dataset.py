"""Build the Challenge 01 benchmark dataset (100 examples).

Design:
- 50 hand-written base items, each with a fictional context, a question and
  2 faithful answer claims written as paraphrases (not copies) of the context.
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
OUT = REPO / "datasets" / "hallucination_qa" / "v1.jsonl"

# (domain, context, question, [faithful claims], (type, index | None for append, text))
BASE = [
    ("company", "Orion Robotics was founded in 2014 by Maria Chen in Pittsburgh. The company builds warehouse picking robots. In 2022 it employed 340 people and opened a second office in Austin.",
     "Who founded Orion Robotics and where?",
     ["Orion Robotics was started by Maria Chen.", "The company was founded in Pittsburgh in 2014."],
     ("entity_swap", 0, "Orion Robotics was started by David Park.")),
    ("product", "The Kestrel X2 air purifier has a HEPA-13 filter that should be replaced every 9 months. It covers rooms up to 45 square meters. The filter indicator light turns red when replacement is due.",
     "How often should the Kestrel X2 filter be replaced?",
     ["The Kestrel X2 filter should be swapped out every 9 months.", "A red indicator light shows when the filter is due."],
     ("number_change", 0, "The Kestrel X2 filter should be swapped out every 6 months.")),
    ("hr_policy", "Employees at Brightline Logistics accrue 1.5 days of paid leave per month. Unused leave of up to 10 days can be carried into the next year. Leave requests must be submitted at least two weeks in advance.",
     "How much unused leave can Brightline employees carry over?",
     ["Brightline employees can carry up to 10 unused leave days into the following year.", "Leave is accrued at 1.5 days per month."],
     ("unsupported_addition", None, "Any carried-over days expire if not used by March 31.")),
    ("medical", "Zelvanor is approved for treating moderate plaque psoriasis in adults. The recommended dose is 40 mg injected every two weeks. Common side effects include injection site redness and headache. It is not approved for patients under 18.",
     "Is Zelvanor approved for children?",
     ["No, Zelvanor is not approved for patients younger than 18.", "It is approved for adults with moderate plaque psoriasis."],
     ("negation", 0, "Yes, Zelvanor is approved for patients younger than 18.")),
    ("history", "The town of Elmsworth was granted a charter in 1783. Its first mayor, Thomas Reed, oversaw the construction of the stone bridge over the Calder River in 1790. The bridge was rebuilt after a flood in 1856.",
     "When was the stone bridge in Elmsworth built?",
     ["The stone bridge over the Calder River was built in 1790.", "Its construction was overseen by Thomas Reed, the town's first mayor."],
     ("date_change", 0, "The stone bridge over the Calder River was built in 1783.")),
    ("software", "Version 3.2 of the Lumen SDK deprecates the sync_upload method. Developers should migrate to upload_async, which supports retries. The sync_upload method will be removed in version 4.0.",
     "What should developers use instead of sync_upload?",
     ["Developers should switch to upload_async.", "The upload_async method supports retries."],
     ("unsupported_addition", None, "The upload_async method is also twice as fast as sync_upload.")),
    ("science", "The Varn Glacier has retreated 2.3 kilometers since 1950, according to a 2021 survey. The survey team was led by the Nordic Ice Institute. Meltwater from the glacier feeds Lake Osk.",
     "How far has the Varn Glacier retreated?",
     ["The Varn Glacier has pulled back 2.3 kilometers since 1950.", "This figure comes from a 2021 survey."],
     ("number_change", 0, "The Varn Glacier has pulled back 3.2 kilometers since 1950.")),
    ("hospitality", "Casa Lucia is open Tuesday through Sunday from 5 pm to 11 pm. It is closed on Mondays. Reservations are required for groups of six or more.",
     "Is Casa Lucia open on Mondays?",
     ["No, Casa Lucia is closed on Mondays.", "It opens Tuesday through Sunday, 5 pm to 11 pm."],
     ("negation", 0, "Yes, Casa Lucia is open on Mondays.")),
    ("finance", "Northwind Bank's savings account pays 3.1% annual interest on balances above $5,000. Balances below $5,000 earn 1.2%. There is no monthly maintenance fee.",
     "What interest does Northwind pay on balances above $5,000?",
     ["Balances over $5,000 earn 3.1% per year.", "The account has no monthly maintenance fee."],
     ("in_context_swap", 0, "Balances over $5,000 earn 1.2% per year.")),
    ("sports", "The Harbor City Gulls won the 2019 regional championship, defeating the Ridgeview Hawks 3-1. Striker Ana Silva scored two of the goals. The team's coach was Peter Novak.",
     "Who scored two goals in the 2019 final?",
     ["Ana Silva scored twice in the final.", "The Gulls beat the Ridgeview Hawks 3-1."],
     ("in_context_swap", 0, "Peter Novak scored twice in the final.")),
    ("travel", "The Alder Line train runs between Fenwick and Port Mara every 30 minutes on weekdays. On weekends it runs hourly. A single ticket costs 4.50 euros.",
     "How often does the Alder Line run on weekends?",
     ["On weekends the Alder Line runs once an hour.", "On weekdays trains come every 30 minutes."],
     ("number_change", 1, "On weekdays trains come every 15 minutes.")),
    ("education", "Halden University requires a minimum GPA of 3.0 for admission to its master's program in data science. Applicants must also submit two letters of recommendation. The program lasts 18 months.",
     "How long is Halden's data science master's program?",
     ["The program takes 18 months to complete.", "Applicants need two recommendation letters."],
     ("unsupported_addition", None, "It can also be completed fully online.")),
    ("hardware", "The Pico-7 microcontroller runs at 133 MHz and has 264 KB of RAM. It supports USB 1.1 and has 26 GPIO pins.",
     "How much RAM does the Pico-7 have?",
     ["The Pico-7 includes 264 KB of RAM.", "It runs at a clock speed of 133 MHz."],
     ("number_change", 0, "The Pico-7 includes 512 KB of RAM.")),
    ("legal", "Under the Riverton tenancy ordinance, landlords must give tenants 60 days' written notice before a rent increase. Increases are capped at 5% per year. The ordinance applies to buildings with four or more units.",
     "How much notice must Riverton landlords give before raising rent?",
     ["Landlords must give 60 days of written notice.", "Annual increases are limited to 5%."],
     ("overgeneralization", 1, "Annual increases are limited to 5% for every rental property in Riverton.")),
    ("device", "The CardioTrack band measures heart rate every 5 seconds and stores 14 days of data. It is water resistant to 50 meters. It is not a medical device and should not be used to diagnose conditions.",
     "Can CardioTrack be used to diagnose heart conditions?",
     ["No, CardioTrack should not be used for diagnosis.", "It is not classified as a medical device."],
     ("negation", 1, "It is a certified medical device.")),
    ("company", "Helix Foods acquired GreenRoot Farms in March 2020 for $48 million. GreenRoot's founder, Lila Okafor, joined Helix as head of sourcing.",
     "How much did Helix Foods pay for GreenRoot Farms?",
     ["Helix Foods paid $48 million for GreenRoot Farms.", "The deal closed in March 2020."],
     ("date_change", 1, "The deal closed in March 2021.")),
    ("space", "The probe Tessera launched in 2031 and reached the asteroid Kallos in 2034. It collected 120 grams of surface material. The sample capsule is scheduled to return to Earth in 2037.",
     "How much material did Tessera collect?",
     ["Tessera gathered 120 grams of material from the surface of Kallos.", "It arrived at Kallos in 2034."],
     ("in_context_swap", 1, "It arrived at Kallos in 2031.")),
    ("support", "Orders from Pinewood Outfitters ship within 2 business days. Standard shipping is free on orders over $75. Returns are accepted within 30 days if items are unworn.",
     "What is Pinewood's return window?",
     ["Pinewood accepts returns within 30 days.", "Items must be unworn to be returned."],
     ("unsupported_addition", None, "Return shipping is always free.")),
    ("agriculture", "The Sunvale tomato variety matures in 68 days and is resistant to fusarium wilt. It grows best in full sun and well-drained soil.",
     "How long does the Sunvale tomato take to mature?",
     ["Sunvale tomatoes mature in about 68 days.", "The variety resists fusarium wilt."],
     ("entity_swap", 1, "The variety resists late blight.")),
    ("hr_policy", "Quanta Labs offers 16 weeks of paid parental leave to all employees after six months of service. Employees can also request a phased return of up to four weeks.",
     "How much paid parental leave does Quanta Labs offer?",
     ["Quanta Labs provides 16 weeks of paid parental leave.", "Employees qualify after six months of service."],
     ("overgeneralization", 1, "Every employee qualifies from their first day.")),
    ("energy", "The Brennan wind farm has 42 turbines and a capacity of 150 megawatts. It began operating in 2018 and supplies power to about 60,000 homes.",
     "How many homes does the Brennan wind farm power?",
     ["It supplies electricity to roughly 60,000 homes.", "The farm has a capacity of 150 megawatts."],
     ("number_change", 0, "It supplies electricity to roughly 90,000 homes.")),
    ("museum", "The Calloway Museum's east wing houses its collection of Byzantine coins. The west wing is dedicated to modern sculpture. Admission is free on the first Sunday of each month.",
     "What is in the Calloway Museum's east wing?",
     ["The east wing holds the museum's Byzantine coin collection.", "Entry is free on the first Sunday of every month."],
     ("in_context_swap", 0, "The east wing holds the museum's modern sculpture collection.")),
    ("medical", "Doravex should be taken with food to reduce stomach upset. Patients should avoid grapefruit juice while taking it, because it can raise drug levels.",
     "Should Doravex be taken with food?",
     ["Yes, Doravex should be taken with food.", "Grapefruit juice should be avoided because it can raise drug levels."],
     ("unsupported_addition", None, "Alcohol should also be avoided for 48 hours after each dose.")),
    ("climate", "Average rainfall in the Merrow Valley is 640 mm per year, most of it falling between November and February. Summers are dry, with less than 20 mm of rain per month.",
     "When does most rain fall in the Merrow Valley?",
     ["Most rain falls from November to February.", "The valley gets about 640 mm of rain a year."],
     ("entity_swap", 0, "Most rain falls from June to August.")),
    ("automotive", "The Vireo EV has a 77 kWh battery and an estimated range of 480 km. It can charge from 10% to 80% in 28 minutes on a 150 kW fast charger.",
     "What is the Vireo EV's range?",
     ["The Vireo EV's estimated range is 480 km.", "It has a 77 kWh battery."],
     ("number_change", 0, "The Vireo EV's estimated range is 520 km.")),
    ("api_docs", "The Atlas API rate limit is 100 requests per minute per key. Exceeding the limit returns HTTP 429. Enterprise plans can request a limit of up to 1,000 requests per minute.",
     "What happens when you exceed the Atlas API rate limit?",
     ["Going over the limit returns an HTTP 429 error.", "The default limit is 100 requests per minute per key."],
     ("unsupported_addition", None, "The key is suspended for 24 hours after three violations.")),
    ("biography", "Painter Irena Volk was born in Gdansk in 1921 and moved to Paris in 1948. Her best-known work, Harbor at Dusk, is held by the Musee Delorme.",
     "Where is Harbor at Dusk held?",
     ["Harbor at Dusk is in the collection of the Musee Delorme.", "Irena Volk moved to Paris in 1948."],
     ("entity_swap", 0, "Harbor at Dusk is in the collection of the Louvre.")),
    ("security", "Acme Corp passwords must be at least 14 characters long and are rotated every 180 days. Multi-factor authentication is required for remote access.",
     "How long must Acme passwords be?",
     ["Passwords at Acme must have at least 14 characters.", "MFA is required for remote access."],
     ("overgeneralization", 1, "MFA is required for every login, including on-site access.")),
    ("urban", "The Lindell Park redevelopment will add 3 hectares of green space and a 400-meter running track. Construction starts in April and is expected to take 14 months.",
     "How long will the Lindell Park construction take?",
     ["Construction is expected to last 14 months.", "It will begin in April."],
     ("date_change", 1, "It will begin in September.")),
    ("nutrition", "A 100 g serving of Brand K oat bars contains 12 g of protein and 8 g of sugar. The bars are gluten-free but are made in a facility that processes nuts.",
     "Are Brand K oat bars safe for people with nut allergies?",
     ["They are made in a facility that also processes nuts.", "The bars are gluten-free."],
     ("negation", 0, "They are made in a nut-free facility.")),
    ("telecom", "Skyreach's Basic plan includes 20 GB of data per month at $25. Unused data does not roll over. Calls within the country are unlimited.",
     "Does unused data roll over on Skyreach Basic?",
     ["No, unused data does not carry over.", "The Basic plan includes 20 GB per month for $25."],
     ("negation", 0, "Yes, unused data rolls over to the next month.")),
    ("archaeology", "Excavations at Tel Sarid in 2016 uncovered a granary dating to around 1200 BCE. The dig was directed by Dr. Hana Levi of Carmel University.",
     "Who directed the Tel Sarid excavation?",
     ["The excavation was led by Dr. Hana Levi.", "She is from Carmel University."],
     ("entity_swap", 1, "She is from the Haifa Institute of Technology.")),
    ("manufacturing", "Line 4 at the Dorsey plant produces 1,200 units per shift. Each shift lasts 8 hours, and the line runs two shifts per day. Planned maintenance happens every Sunday.",
     "How many units does Line 4 produce per day?",
     ["Line 4 makes 1,200 units per shift.", "It runs two shifts a day, so about 2,400 units daily."],
     ("number_change", 1, "It runs three shifts a day, so about 3,600 units daily.")),
    ("software", "Kite Editor 5.0 adds real-time collaboration and a dark theme. Plugins written for version 4 must be updated to the new API before they will load.",
     "Do version 4 plugins work in Kite Editor 5.0?",
     ["No, version 4 plugins need to be updated to the new API first.", "Version 5.0 also adds a dark theme."],
     ("unsupported_addition", None, "An automatic migration tool converts old plugins.")),
    ("zoology", "The Alpine mole vole lives at elevations between 1,800 and 2,600 meters. It feeds mainly on roots and bulbs and is active year-round.",
     "Does the Alpine mole vole hibernate?",
     ["No, it stays active all year.", "It eats mostly roots and bulbs."],
     ("negation", 0, "Yes, it hibernates through the winter.")),
    ("insurance", "The Silver plan covers 70% of in-network hospital costs after a $1,500 deductible. Out-of-network care is not covered except in emergencies.",
     "Does the Silver plan cover out-of-network care?",
     ["Only in emergencies; otherwise out-of-network care is not covered.", "In-network hospital costs are covered at 70% after the deductible."],
     ("overgeneralization", 0, "Yes, out-of-network care is covered at the same 70% rate.")),
    ("gaming", "Starfall Online's winter update adds a new region called the Frostmarch and raises the level cap from 60 to 70. The update releases on December 12.",
     "What is the new level cap in Starfall Online?",
     ["The level cap goes up to 70.", "The update also adds the Frostmarch region."],
     ("in_context_swap", 0, "The level cap goes up to 60.")),
    ("utility", "Residents of Ashby are limited to watering lawns on Tuesdays and Saturdays during the summer restriction period. Fines start at $100 for a first violation.",
     "What is the fine for a first watering violation in Ashby?",
     ["A first violation carries a fine starting at $100.", "Lawn watering is allowed on Tuesdays and Saturdays."],
     ("entity_swap", 1, "Lawn watering is allowed on Mondays and Thursdays.")),
    ("research", "The study enrolled 412 participants and found that the intervention reduced average sleep onset time by 11 minutes. The authors note that the effect was not significant in participants over 65.",
     "Did the intervention work for people over 65?",
     ["The effect was not significant for participants older than 65.", "Overall, sleep onset time dropped by 11 minutes on average."],
     ("overgeneralization", 0, "It worked equally well across all age groups.")),
    ("logistics", "Containers arriving at Port Halvard clear customs in an average of 36 hours. Refrigerated containers are prioritized and typically clear within 12 hours.",
     "How quickly do refrigerated containers clear customs at Port Halvard?",
     ["Refrigerated containers usually clear within 12 hours.", "They are given priority."],
     ("in_context_swap", 0, "Refrigerated containers usually clear within 36 hours.")),
    ("library", "The Fairmont Public Library lends up to 15 items per card for three weeks. Items can be renewed twice unless another patron has placed a hold.",
     "How many times can items be renewed?",
     ["Items can be renewed two times.", "Renewal is not possible if someone else has a hold on the item."],
     ("number_change", 0, "Items can be renewed five times.")),
    ("cloud", "Nimbus Storage replicates data across three availability zones in a region. Cross-region replication is available as an add-on for $0.02 per GB.",
     "Does Nimbus replicate data across regions by default?",
     ["No, cross-region replication is a paid add-on.", "By default data is copied across three availability zones within a region."],
     ("unsupported_addition", None, "Cross-region replication completes within 15 minutes.")),
    ("events", "The Lakeside Jazz Festival runs from July 18 to July 21 at Marlow Pier. Tickets for a single day cost $45, and a full pass costs $150.",
     "How much is a full festival pass?",
     ["A full pass costs $150.", "The festival takes place at Marlow Pier."],
     ("entity_swap", 1, "The festival takes place at Riverside Park.")),
    ("botany", "The ghost orchid flowers only once per year, usually in midsummer, and each bloom lasts about two weeks. It has no leaves and photosynthesizes through its roots.",
     "How does the ghost orchid photosynthesize?",
     ["It photosynthesizes through its roots, since it has no leaves.", "Each bloom lasts around two weeks."],
     ("unsupported_addition", None, "It is pollinated exclusively by the giant sphinx moth.")),
    ("payments", "Refunds through PayLane are returned to the original payment method within 5 to 7 business days. Refunds for purchases older than 90 days require manual review.",
     "How long do PayLane refunds take?",
     ["Refunds arrive in 5 to 7 business days.", "They go back to the original payment method."],
     ("number_change", 0, "Refunds arrive in 1 to 2 business days.")),
    ("government", "The Kelburn city council has 11 members elected for four-year terms. Elections are staggered so that roughly half the seats are contested every two years.",
     "How long is a Kelburn council term?",
     ["Council members serve four-year terms.", "About half the seats are up for election every two years."],
     ("in_context_swap", 0, "Council members serve two-year terms.")),
    ("retail", "The Mesa store's holiday hours are 8 am to 10 pm from December 1 to December 23. The store closes at 4 pm on December 24 and is closed on December 25.",
     "Is the Mesa store open on December 25?",
     ["No, the store is closed on December 25.", "On December 24 it closes at 4 pm."],
     ("negation", 0, "Yes, the store is open from 8 am to 10 pm on December 25.")),
    ("chemistry", "Compound TX-9 is stable at room temperature but decomposes above 180 C. It is soluble in ethanol and insoluble in water.",
     "Is TX-9 soluble in water?",
     ["No, TX-9 does not dissolve in water.", "It is soluble in ethanol."],
     ("entity_swap", 1, "It is soluble in acetone.")),
    ("transport", "Bikes are allowed on Metroline trains outside peak hours, which are 7 to 9 am and 4 to 6 pm on weekdays. Folding bikes are allowed at all times.",
     "Can I bring a regular bike on Metroline at 8 am on a weekday?",
     ["No, 8 am on a weekday is within peak hours, when regular bikes are not allowed.", "Folding bikes are permitted at any time."],
     ("overgeneralization", 1, "All bikes are permitted at any time.")),
    ("ai", "The Wren-2 language model was trained on 1.4 trillion tokens and has a context window of 32,000 tokens. It was released under the Apache 2.0 license in May 2025.",
     "What license is Wren-2 released under?",
     ["Wren-2 is released under the Apache 2.0 license.", "It has a 32,000-token context window."],
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
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} examples to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
