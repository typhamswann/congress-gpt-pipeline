import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from ast import literal_eval
import os
import openai

api_key = '<api key>'

openai.api_key = api_key

def classify_claim(claim):
    raw = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {
        "role": "system",
        "content": "The claim taxonomy is as follows:\n•Global warming is not happening\n•Ice/permafrost/snow cover isn't melting\n•Antarctica is gaining ice/not warming\n•Greenland is gaining ice/not melting\n•Arctic sea ice isn't vanishing\n•Glaciers aren't vanishing\n•We're heading into an ice age/global cooling\n•Weather is cold/snowing\n•Climate hasn't warmed/changed over the last (few) decade(s)\n•Oceans are cooling/not warming\n•Sea level rise is exaggerated/not accelerating\n•Extreme weather isn't increasing/has happened before/isn't linked to climate change\n•They changed the name from 'global warming' to 'climate change'\n•Human greenhouse gases are not causing climate change\n•It's natural cycles/variation\n•It's the sun/cosmic rays/astronomical\n•It's geological (includes volcanoes)\n•It's the ocean/internal variability\n•Climate has changed naturally/been warm in the past\n•Human CO2 emissions are tiny compared to natural CO2 emission\n•It's the clouds\n•It's non-greenhouse gas human climate forcings (aerosols, land use)\n•There's no evidence for greenhouse effect/carbon dioxide driving climate change\n•Carbon dioxide is just a trace gas\n•Greenhouse effect is saturated/logarithmic\n•Carbon dioxide lags/not correlated with climate change\n•Water vapor is the most powerful greenhouse gas/It's the clouds\n•There's no tropospheric hot spot\n•CO2 was higher in the past\n•CO2 is not rising/ocean pH is not falling\n•Human CO2 emissions are miniscule/not raising atmospheric CO2\n•Climate impacts/global warming is beneficial/not bad\n•Climate sensitivity is low/negative feedbacks reduce warming\n•Species/plants/reefs aren't showing climate impacts yet/are benefiting from climate change\n•Species can adapt to global warming\n•Polar bears are not in danger from climate change\n•Ocean acidification/coral impacts aren't serious\n•CO2 is beneficial/not a pollutant\n•CO2 is plant food\n•It's only a few degrees (or less)\n•Climate change does not contribute to human conflict/threaten national security\n•Climate change doesn't negatively impact health / positive health benefits outweigh negative ones\n•Climate change doesn't negatively impact the economy / warmer climates are associated with prosperity\n•Warmer climates are associated with prosperity\n•Climate movement/science is unreliable\n•Climate-related science is uncertain/unsound/unreliable (data, methods & models)\n•There's no scientific consensus on climate/the science isn't settled\n•Data/proxies are unreliable unreliable (includes hockey stick)\n•Temperature record is unreliable\n•Models are wrong/unreliable/uncertain\n•Climate movement is alarmist/wrong/political/biased/hypocritical (people or groups)\n•Climate movement is religion\n•Media (including bloggers) is alarmist/wrong/political/biased\n•Politicians/government/UN are alarmist/wrong/political/biased\n•Environmentalists are alarmist/wrong/political/biased\n•Scientists/academics are alarmist/wrong/political/biased\n•Climate change (science or policy) is a conspiracy (deception)\n•Climate policy/renewables is a hoax/scam/conspiracy/secretive\n•Climate science is a hoax/scam/conspiracy/secretive/money-motivated (includes climategate)\n•Climate policies are harmful\n•Climate policy will increase costs / harm economy\n•Policy harms competitiveness\n•Climate policy will kill jobs / harm vulnerable members of society\n•Climate policy exacerbates global poverty and famine\n•Proposed action would weaken national security / energy security / national sovereignty / cause conflict\n•Proposed action would actually harm the environment and species\n•Policies create uncertainty/instability / have unpredictable/unintended consequences\n•Climate policy limits liberty / freedom / capitalism\n•Climate policies are ineffective / flawed\n•Clean energy/green jobs/businesses won't work\n•Climate policy will make negligible difference to climate change/only make a difference in the distant future\n•A single country/region only contributes a small % of global emissions\n•Climate action is pointless because of other countries'/China's/India's emissions\n•Existing climate policy solutions don't work (no reason given; often with a concrete policy example)\n•Policy can be gamed/manipulated\n•We should focus on something other than climate change mitigation policies\n•Markets/private sector are economically more efficient than government policies (voluntary policies)\n•Individuals/consumers are responsible/should change their behaviour\n•Future generations will be richer/have cheaper solutions available and better able to solve the problem\n•Better to adapt / geoengineer / increase resiliency\n•We should invest in mitigation technology\n•We should focus on helping other countries reduce their emissions (using technology)\n•We should address other problems first (eg reduce poverty/disease first)\n•Climate policy is too difficult / premature\n•Not ready for policy: we need a perfect policy and we need to know all policy implications before passing a policy (we need more data/models/info/science/technology)\n•It's too hard to solve: Climate policy is politically/legally/economically/technically too difficult\n•It's too late to do anything to mitigate climate change\n•Media/public support/acceptance is low/decreasing\n•Climate-friendly technologies and practices won't work / aren't ready yet / aren't viable yet\n•Climate-friendly technologies/practices are harmful\n•Too expensive / uneconomical\n•Harmful to people / the environment\n•Counterproductive /  increase emissions\n•Climate-friendly technologies/practices are ineffective\n•Not ready\n•Not enough / can't scale / can't provide baseload\n•People need energy (e.g., from fossil fuels)\n•Fossil fuel reserves are plentiful\n•Fossil fuels are cheap/good/safe for economy/society/environment\n•Fossil fuels are important for economic growth / economic development\n•Fossil fuel extraction is important for our energy security\n•Our fossil fuels are cleaner than others\n•Fossil fuels are necessary to cover the energy demand\n•We have the right to profit from fossil fuels like others have\n•No need for more action\n•We are already taking climate change seriously, so there is no need for more action\n•We are already doing enough to address climate change, so there is no need for more action\n•We are already contributing enough to the societal good we don't need to also adress climate change issues\n----------------------------------\n\nYou are a labeler. You will receive a paragraph from the congressional record. Please label it 1.0 if it makes any climate disinformation or delay claims. A climate disinformation or delay claim can contain any of the claims included in the taxonomy or other common disinformation or delay claims. If a claim is not a delay or disinformation claim, which is most common, label it 0.0. Let's think step by step."
        },
        {
        "role": "user",
        "content": "Paragraph: \"according to the climate graph on the right\""
        },
        {
        "role": "assistant",
        "content": "The statement \"according to the climate graph on the right\" does not provide enough information to determine whether it is making a climate disinformation or delay claim. The statement is referring to a climate graph, but without seeing the graph or knowing what it represents, it is impossible to evaluate the accuracy of the information being presented. Therefore, this statement should be labeled as ['0']."
        },
        {
        "role": "user",
        "content": "Paragraph: \"Lets consider a current special case to illustrate these concepts. Dr. Wei-Hock Soon of the Harvard Smithsonian Center for Astrophysics has been regularly listed as one of the bad kids on the tattling websites because of his climate research. The insinuation is that he has accepted bribes from oil companies to take the positions he does. The implied allegation is only bad in his case because the complaining persons dont like his positions. Recently he coauthored a paper that caused them much consternation, igniting a campaign from the same quarters that generate the tattling websites, to get Dr. Soon fired. A letter was written to his employer accusing him of crimes that even Kafka would find hard to follow. But basically they boil down to the usual: taking money for lies and not confessing. Welcome to the inquisition.\""
        },
        {
        "role": "assistant",
        "content": "This paragraph contains several elements that fall under the category of climate disinformation or delay claims. It includes an attack on Dr. Wei-Hock Soon, implying that he has accepted bribes from oil companies to support certain climate positions. It dismisses the allegations against him as mere opposition to his positions and likens it to an inquisition. Furthermore, it suggests that the accusations are based on taking money for lies and not confessing. Therefore, this paragraph should be labeled as ['1']."
        },
        {
        "role": "user",
        "content": "Paragraph: \"In resisting calls for his resignation, Pachauri denied that TERI benefited financially from the inadequate due diligence:\""
        },
        {
        "role": "assistant",
        "content": "This statement does not make any climate disinformation or delay claims. It is simply mentioning that Pachauri denied that TERI (The Energy and Resources Institute) benefited financially from inadequate due diligence. Therefore, this statement should be labeled as ['0']."
        },
        {
        "role": "user",
        "content": f"Paragraph: {claim}"
        }
    ],
    temperature=0,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    print(raw.choices[0].message.content)
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "system",
        "content": "You are a labeler. Please extract the label, either ['1'] or ['0'] from the text."
        },
        {
        "role": "user",
        "content": "The statement \"according to the climate graph on the right\" does not provide enough information to determine whether it is making a climate disinformation or delay claim. The statement is referring to a climate graph, but without seeing the graph or knowing what it represents, it is impossible to evaluate the accuracy of the information being presented. Therefore, this statement should be labeled as ['0']."
        },
        {
        "role": "assistant",
        "content": "['0']"
        },
        {
        "role": "user",
        "content": "This paragraph contains several elements that fall under the category of climate disinformation or delay claims. It includes an attack on Dr. Wei-Hock Soon, implying that he has accepted bribes from oil companies to support certain climate positions. It dismisses the allegations against him as mere opposition to his positions and likens it to an inquisition. Furthermore, it suggests that the accusations are based on taking money for lies and not confessing. Therefore, this paragraph should be labeled as ['1']"
        },
        {
        "role": "assistant",
        "content": "['1']"
        },
        {
        "role": "user",
        "content": "This statement does not make any climate disinformation or delay claims. It is simply mentioning that Pachauri denied that TERI (The Energy and Resources Institute) benefited financially from inadequate due diligence. Therefore, this statement should be labeled as ['0']"
        },
        {
        "role": "assistant",
        "content": "['0']"
        },
        {
        "role": "user",
        "content": raw.choices[0].message.content
        }
    ],
    temperature=0,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    print(response.choices[0].message.content)
    if response.choices[0].message.content != "['0']" and response.choices[0].message.content != "[]" :
        classified = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
            "role": "system",
            "content": "Label Taxonomy: ```\n1.1 - Ice/permafrost/snow cover isn't melting\n1.1 - Antarctica is gaining ice/not warming\n1.1 - Greenland is gaining ice/not melting\n1.1 - Arctic sea ice isn't vanishing\n1.1 - Glaciers aren't vanishing\n1.2 - We're heading into an ice age/global cooling\n1.3 - Weather is cold/snowing\n1.4 - Climate hasn't warmed/changed over the last (few) decade(s)\n1.5 - Oceans are cooling/not warming\n1.6 - Sea level rise is exaggerated/not accelerating\n1.7 - Extreme weather isn't increasing/has happened before/isn't linked to climate change\n1.8 - They changed the name from 'global warming' to 'climate change'\n2.1 - It's natural cycles/variation\n2.1 - It's the sun/cosmic rays/astronomical\n2.1 - It's geological (includes volcanoes)\n2.1 - It's the ocean/internal variability\n2.1 - Climate has changed naturally/been warm in the past\n2.1 - Human CO2 emissions are tiny compared to natural CO2 emission\n2.1 - It's the clouds\n2.2 - It's non-greenhouse gas human climate forcings (aerosols, land use)\n2.3 - There's no evidence for greenhouse effect/carbon dioxide driving climate change\n2.3 - Carbon dioxide is just a trace gas\n2.3 - Greenhouse effect is saturated/logarithmic\n2.3 - Carbon dioxide lags/not correlated with climate change\n2.3 - Water vapor is the most powerful greenhouse gas/It's the clouds\n2.3 - There's no tropospheric hot spot\n2.3 - CO2 was higher in the past\n2.4 - CO2 is not rising/ocean pH is not falling\n2.5 - Human CO2 emissions are miniscule/not raising atmospheric CO2\n3.1 - Climate sensitivity is low/negative feedbacks reduce warming\n3.2 - Species/plants/reefs aren't showing climate impacts yet/are benefiting from climate change\n3.2 - Species can adapt to global warming\n3.2 - Polar bears are not in danger from climate change\n3.2 - Ocean acidification/coral impacts aren't serious\n3.3 - CO2 is beneficial/not a pollutant\n3.3 - CO2 is plant food\n3.4 - It's only a few degrees (or less)\n3.5 - Climate change does not contribute to human conflict/threaten national security\n3.6 - Climate change doesn't negatively impact health / positive health benefits outweigh negative ones\n3.7 - Climate change doesn't negatively impact the economy / warmer climates are associated with prosperity\n3.8 - Warmer climates are associated with prosperity\n4.1 - Climate policies are harmful\n4.1 - Climate policy will increase costs / harm economy\n4.1 - Policy harms competitiveness\n4.1 - Climate policy will kill jobs / harm vulnerable members of society\n4.1 - Climate policy exacerbates global poverty and famine\n4.1 - Proposed action would weaken national security / energy security / national sovereignty / cause conflict\n4.1 - Proposed action would actually harm the environment and species\n4.1 - Policies create uncertainty/instability / have unpredictable/unintended consequences\n4.1 - Climate policy limits liberty / freedom / capitalism\n4.2 - Climate policies are ineffective / flawed\n4.2 - Clean energy/green jobs/businesses won't work\n4.2 - Climate policy will make negligible difference to climate change/only make a difference in the distant future\n4.2 - A single country/region only contributes a small % of global emissions\n4.2 - Climate action is pointless because of other countries'/China's/India's emissions\n4.2 - Existing climate policy solutions don't work (no reason given; often with a concrete policy example)\n4.2 - Policy can be gamed/manipulated\t\n4.3 - We should focus on something other than climate change mitigation policies\n4.3 - Markets/private sector are economically more efficient than government policies (voluntary policies)\n4.3 - Individuals/consumers are responsible/should change their behaviour\n4.3 - Future generations will be richer/have cheaper solutions available and better able to solve the problem\n4.3 - Technofix\n4.3 - Better to adapt / geoengineer / increase resiliency\n4.3 - We should invest in mitigation technology\n4.3 - We should focus on helping other countries reduce their emissions (using technology)\n4.3 - We should address other problems first (eg reduce poverty/disease first)\t\n4.4 - Climate policy is too difficult / premature\n4.4 - Not ready for policy: we need a perfect policy and we need to know all policy implications before passing a policy (we need more data/models/info/science/technology)\n4.4 - It's too hard to solve: Climate policy is politically/legally/economically/technically too difficult\n4.4 - It's too late to do anything to mitigate climate change\n4.4 - Media/public support/acceptance is low/decreasing\n4.5 - Climate-friendly technologies and practices won't work / aren't ready yet / aren't viable yet\n4.5 - Climate-friendly technologies/practices are harmful\n4.5 - Too expensive / uneconomical\n4.5 - Harmful to people / the environment\n4.5 - Counterproductive /  increase emissions\n4.5 - Climate-friendly technologies/practices are ineffective\n4.5 - Not ready\n4.5 - Not enough / can't scale / can't provide baseload\n\t\n4.6 - People need energy (e.g., from fossil fuels)\n4.6 - Fossil fuel reserves are plentiful\n4.6 - Fossil fuels are cheap/good/safe for economy/society/environment\n4.6 - Fossil fuels are important for economic growth / economic development\n4.6 - Fossil fuel extraction is important for our energy security\n4.6 - Our fossil fuels are cleaner than others\n4.6 - Fossil fuels are necessary to cover the energy demand\n4.6 - We have the right to profit from fossil fuels like others have\n4.7 - No need for more action\n4.7 - We are already taking climate change seriously, so there is no need for more action\n4.7 - We are already doing enough to address climate change, so there is no need for more action\n4.7 - We are already contributing enough to the societal good we don't need to also adress climate change issues\n5.1 - Climate-related science is uncertain/unsound/unreliable (data, methods & models)\n5.1 - There's no scientific consensus on climate/the science isn't settled\n5.1 - Data/proxies are unreliable unreliable (includes hockey stick)\n5.1 - Temperature record is unreliable\n5.1 - Models are wrong/unreliable/uncertain\n5.2 - Climate movement is alarmist/wrong/political/biased/hypocritical (people or groups)\n5.2 - Climate movement is religion\n5.2 - Media (including bloggers) is alarmist/wrong/political/biased\n5.2 - Politicians/government/UN are alarmist/wrong/political/biased\n5.2 - Environmentalists are alarmist/wrong/political/biased\n5.2 - Scientists/academics are alarmist/wrong/political/biased\n5.3 - Climate change (science or policy) is a conspiracy (deception)\n5.3 - Climate policy/renewables is a hoax/scam/conspiracy/secretive\n5.3 - Climate science is a hoax/scam/conspiracy/secretive/money-motivated (includes climategate)\n```\n------------------------------------------\nYou are a human labeler tasked with labeling climate disinformation and delay discourses. You will be shown a claim from the congressional record. Your job is to identify which label best matches the claim. There may be multiple claims, in which case you should label multiple, though this occurs rarely. Let's think step by step.\n\n"
            },
            {
            "role": "user",
            "content": "Claim: \"Punitive Oil and Gas Taxes--Businesses claim that Congress must be \nmindful of the crosswinds hitting the American economy from the \nfinancial sector to the housing sectors. Many believe tax increases on \nthe oil and gas industries are out of sync with an American economy \nshowing great demand for increased domestic energy production, which \ncould provide the opportunity for the energy industry to add a \nsignificant number of high-wage jobs. Many are concerned with \nprovisions that would freeze the section 199 deduction for oil and gas \ncompanies. This change would discourage energy investment, resulting in \nthe loss of jobs, a decrease in the supply of oil and gas, and an \nincrease in the costs for businesses that rely on oil and gas.\"\n\nLet's think step by step"
            },
            {
            "role": "assistant",
            "content": "The claim touches on several elements related to the impact of policy on the economy, the oil and gas industry, and potential job loss. Here is the step-by-step analysis to match the appropriate label(s):\n\n1. Understanding the Claim Context: The claim comes from discussions about the effect of taxing the oil and gas industry. It suggests that such taxes would be harmful to the economy, particularly in terms of energy production, job creation, and business costs.\n\n2. Identifying the Core Arguments:\n   - Tax increases on oil and gas are harmful to the economy.\n   - Freezing tax deductions for oil and gas companies will discourage investment in the energy sector.\n   - Such policy changes will result in job losses, decreased oil and gas supplies, and higher costs for businesses dependent on these resources.\n\n3. Matching the Claim with the Label Taxonomy:\n   - The argument against tax increases could be considered an argument against climate policy, as these taxes are often proposed as a means to reduce greenhouse gas emissions.\n   - The emphasis on the potential economic downsides (loss of jobs, higher costs, less investment) is a common argument used to push back against climate policies.\n\nGiven the elements in the claim, the most fitting labels appear to be:\n\n- 4.1 - Climate policy will increase costs / harm economy: The claim argues that tax increases on the oil and gas industry will result in higher costs for businesses.\n- 4.1 - Climate policy will kill jobs / harm vulnerable members of society: It's suggested that the policy will result in job losses.\n- 4.1 - Proposed action would actually harm the environment and species: While not directly stated, the argument implies that a decrease in the domestic supply of oil and gas could lead to increased reliance on potentially less environmentally friendly sources.\n- 4.6 - Fossil fuel reserves are plentiful: The claim implicitly suggests that domestic energy production should be increased to meet demand, which hints at the notion that there is an abundance of fossil fuel reserves.\n- 4.6 - Fossil fuels are important for economic growth / economic development: The claim states that the oil and gas industry could add high-wage jobs, suggesting its importance for economic growth.\n\nThese labels most closely match the claim as it presents a viewpoint that is focused on the economic aspects and potential negative impacts of climate-related policies on the oil and gas industries."
            },
            {
            "role": "user",
            "content": "Claim: \"Mr. INHOFE. Madam President, I am returning to the floor, as I have \nmany times in the last few years, to further address what I have \nconsidered to be probably the greatest single hoax ever perpetrated on \nthe American people, and that is this thing called global warming. As I \nnoted in my last speech, there is a perception, especially among the \nmedia and the environmental elitists, that the scientific community has \nreached a consensus on global warming. As Sir David King, the chief \nscience adviser to the British Government, recently said:\"\n\nLet's think step by step."
            },
            {
            "role": "assistant",
            "content": "The claim includes arguments on the existence of global warming and the perceived consensus in the scientific community. Here's the analysis step-by-step:\n\n1. Understanding the Claim Context: The claim suggests skepticism about the validity of the phenomenon known as global warming, considering it to be a \"hoax\".\n\n2. Identifying the Core Arguments: \n   - Global warming is claimed to be a hoax.\n   - The idea that there's a consensus among scientists on global warming is challenged.\n\n3. Matching the Claim with the Label Taxonomy:\n   - The claim focuses on the denial of global warming as a valid scientific concept as well as a challenge to the claim of consensus within the scientific community. \n   \nBased on this, the claim can be labelled with:\n\n- 5.3 - Climate science is a hoax/scam/conspiracy/secretive/money-motivated: The use of the term \"hoax\" directly aligns with this category.\n- 5.1 - There's no scientific consensus on climate/the science isn't settled: The speaker questions the idea of a consensus in the scientific community on global warming.\n- 5.2 - Environmentalists are alarmist/wrong/political/biased\n \nThese labels most closely fit with the claim, as it presents a viewpoint focused on skepticism towards global warming and challenges the idea of scientific"
            },
            {
            "role": "user",
            "content": f"Claim: {claim} \n\nLet's think step by step."
            }
        ],
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
        print(classified.choices[0].message.content)
        final_extracted = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
            "role": "system",
            "content": "You are a labeler. Please extract the label, either the label or labels from the text in list form. "
            },
            {
            "role": "user",
            "content": "\"The claim touches on several elements related to the impact of policy on the economy, the oil and gas industry, and potential job loss. Here is the step-by-step analysis to match the appropriate label(s):\n\n1. Understanding the Claim Context: The claim comes from discussions about the effect of taxing the oil and gas industry. It suggests that such taxes would be harmful to the economy, particularly in terms of energy production, job creation, and business costs.\n\n2. Identifying the Core Arguments:\n \n- Tax increases on oil and gas are harmful to the economy.\n- Freezing tax deductions for oil and gas companies will discourage investment in the energy sector.\n- Such policy changes will result in job losses, decreased oil and gas supplies, and higher costs for businesses dependent on these resources.\n\n3. Matching the Claim with the Label Taxonomy:\n- The argument against tax increases could be considered an argument against climate policy, as these taxes are often proposed as a means to reduce greenhouse gas emissions.\n- The emphasis on the potential economic downsides (loss of jobs, higher costs, less investment) is a common argument used to push back against climate policies.\n\nGiven the elements in the claim, the most fitting labels appear to be:\n\n- 4.1 - Climate policy will increase costs / harm economy: The claim argues that tax increases on the oil and gas industry will result in higher costs for businesses.\n- 4.1 - Climate policy will kill jobs / harm vulnerable members of society: It's suggested that the policy will result in job losses.\n- 4.1 - Proposed action would actually harm the environment and species: While not directly stated, the argument implies that a decrease in the domestic supply of oil and gas could lead to increased reliance on potentially less environmentally friendly sources.\n4.6 - Fossil fuel reserves are plentiful: The claim implicitly suggests that domestic energy production should be increased to meet demand, which hints at the notion that there is an abundance of fossil fuel reserves.\n4.6 - Fossil fuels are important for economic growth / economic development: The claim states that the oil and gas industry could add high-wage jobs, suggesting its importance for economic growth.\nThese labels most closely match the claim as it presents a viewpoint that is focused on the economic aspects and potential negative impacts of climate-related policies on the oil and gas industries.\""
            },
            {
            "role": "assistant",
            "content": "['4.1', '4.6']"
            },
            {
            "role": "user",
            "content": "\"The claim includes arguments on the existence of global warming and the perceived consensus in the scientific community. Here's the analysis step-by-step:\n\n1. Understanding the Claim Context: The claim suggests skepticism about the validity of the phenomenon known as global warming, considering it to be a \"hoax\".\n\n2. Identifying the Core Arguments: \n- Global warming is claimed to be a hoax.\n- The idea that there's a consensus among scientists on global warming is challenged.\n\n3. Matching the Claim with the Label Taxonomy:\n- The claim focuses on the denial of global warming as a valid scientific concept as well as a challenge to the claim of consensus within the scientific community.\n\nBased on this, the claim can be labelled with:\n\n- 5.3 - Climate science is a hoax/scam/conspiracy/secretive/money-motivated: The use of the term \"hoax\" directly aligns with this category.\n- 5.1 - There's no scientific consensus on climate/the science isn't settled: The speaker questions the idea of a consensus in the scientific community on global warming.\n- 5.2 - Environmentalists are alarmist/wrong/political/biased\n\nThese labels most closely fit with the claim, as it presents a viewpoint focused on skepticism towards global warming and challenges the idea of scientific\""
            },
            {
            "role": "assistant",
            "content": "['5.3', '5.1', 5.2']"
            },
            {
            "role": "user",
            "content": classified.choices[0].message.content
            }
        ],
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
        print(final_extracted.choices[0].message.content)
        return final_extracted.choices[0].message.content
    else: 
        return response.choices[0].message.content

# Load the CSV file
df = pd.read_csv('<congress data csv>')
df['claims_list'] = df['claims_list'].apply(literal_eval)
df['claims_list'] = df['claims_list'].apply(sorted)

# Initialize MultiLabelBinarizer
mlb = MultiLabelBinarizer()

# Fit the binarizer and transform the ground truth labels
y_true = mlb.fit_transform(df['claims_list'])

# Initialize list to keep track of incorrect predictions and the binary format of predicted labels
incorrect_predictions = []
y_pred = []

# Classify each claim and track progress
for index, row in df.iterrows():
    # Get the predicted labels as a list and sort them
    predicted_labels = literal_eval(classify_claim(row['text']))
    predicted_labels_sorted = sorted(predicted_labels)

    # Transform the predicted labels to binary format
    y_pred_row = mlb.transform([predicted_labels_sorted])[0]
    y_pred.append(y_pred_row)

    # Check if the prediction is correct (in binary format)
    if not (y_pred_row == y_true[index]).all():
        # If incorrect, append to the list for CSV export
        incorrect_predictions.append({
            'claim': row['text'],
            'predicted': predicted_labels_sorted,
            'ground_truth': row['claims_list']
        })

    # Print progress
    print(f"Processed {index + 1}/{len(df)} claims", end='\r')

# Calculate the metrics using the binary arrays
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')

# Output the performance metrics
print(f'\nAccuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')

# Convert incorrect predictions to a DataFrame and save as CSV
incorrect_df = pd.DataFrame(incorrect_predictions)
incorrect_df.to_csv('incorrect_predictions.csv', index=False)

print("Incorrect predictions have been saved to 'incorrect_predictions.csv'.")
