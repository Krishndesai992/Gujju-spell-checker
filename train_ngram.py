"""
train_ngram.py - Training Script for Statistical N-Gram Language Model

Trains the N-gram model with comprehensive Gujarati vocabulary coverage:
1. Loads training sentences from data/gujarati_training.csv.
2. Incorporates authentic domain sentences covering standard vocabulary.
3. Preprocesses and tokenizes each sentence using preprocessing.py.
4. Builds the vocabulary automatically.
5. Counts unigrams, bigrams, and trigrams.
6. Saves the trained statistical model to model/ngram_model.json.
"""

import os
import sys
import csv
from pathlib import Path
from preprocessing import preprocess
from ngram_model import NGramLanguageModel

# Standard authentic Gujarati sentences ensuring full vocabulary coverage
AUTHENTIC_CORPUS_SENTENCES = [
    # Core target vocabulary sentences (1-89)
    "આપણે મહેમાનનો આદરપૂર્વક આવકાર કરીએ છીએ.",
    "નવા કામ માટે સારો અનુભવ ખૂબ જરૂરી છે.",
    "તેમણે પુસ્તકનો સુંદર ગુજરાતી અનુવાદ કર્યો છે.",
    "જીવનમાં નિયમિત અભ્યાસ કરવો આવશ્યક છે.",
    "રાત્રે ખુલ્લા આકાશમાં અનંત અવકાશ જોવા મળે છે.",
    "બંધારણ દરેક નાગરિકને સમાન અધિકાર આપે છે.",
    "શાળા અને કોલેજમાં વિદ્વાન અધ્યાપક ભણાવે છે.",
    "ઉદ્યોગ અને વેપારથી દેશની અર્થવ્યવસ્થા મજબૂત બને છે.",
    "કોઈપણ કાર્ય સફળ કરવા માટે યોગ્ય આયોજન જરૂરી છે.",
    "સ્વસ્થ જીવનશૈલી અપનાવવાથી મનુષ્યનું આયુષ્ય વધે છે.",
    "શહેરમાં એક ભવ્ય અને ઊંચી ઇમારત બની રહી છે.",
    "દિવાળી અને નવરાત્રી આપણો મોટો ઉત્સવ છે.",
    "શિક્ષકે ગણિતનો દાખલો ગણવા માટે સરળ ઉદાહરણ આપ્યું.",
    "રાજ્યમાં નવો ઉદ્યોગ શરૂ થવાથી રોજગારી વધે છે.",
    "સૌ નાગરિકો વચ્ચે શાંતિ અને એકતા હોવી જોઈએ.",
    "સમાજમાં સારા અને સચ્ચાઈવાળા માણસની સાચી ઓળખ થાય છે.",
    "બીમાર દર્દી માટે યોગ્ય ઔષધ ખૂબ ઉપયોગી છે.",
    "બે દેશો વચ્ચે શાંતિ અને વેપાર માટે કરાર થયો.",
    "સવારે આકાશમાં સુંદર સફેદ કબૂતર ઊડે છે.",
    "દરેક વ્યક્તિ પોતાના કુટુંબ સાથે ખુશીથી રહે છે.",
    "સુંદર કુદરત માનવજીવનને શાંતિ અને આનંદ આપે છે.",
    "આજના યુગમાં કમ્પ્યુટરનું નવું કૌશલ્ય શીખવું જરૂરી છે.",
    "બજારમાં ઉત્તમ ગુણવત્તાવાળા પુસ્તકની વ્યાજબી કિંમત છે.",
    "સવારે સૂર્યનું નવું કિરણ સોનેરી પ્રકાશ લાવે છે.",
    "ખેતરમાં મહેનતુ કિસાન અનાજ પકવે છે.",
    "દેશમાં કાયદો અને નિયમનું પાલન થવું જોઈએ.",
    "જમીન ખરીદવા માટે કાયદાકીય સલાહ લેવી પડે છે.",
    "ખેડૂત વહેલી સવારે પોતાના ખેતરમાં જાય છે.",
    "મેદાનમાં બાળકો ઉત્સાહથી નવો ખેલ રમે છે.",
    "વેપારી પોતાના નફા અને નુકસાનની ગણતરી કરે છે.",
    "સખત મહેનત અને શિક્ષણથી ગરીબી દૂર થાય છે.",
    "આપણા દેશ અને સંસ્કૃતિ પર આપણને ઘણું ગૌરવ છે.",
    "વિજ્ઞાનની નવી શોધો માનવજાત માટે ચમત્કાર સમાન છે.",
    "મુશ્કેલ પ્રશ્નોનો ઉકેલ લાવવા માટે ગહન ચિંતન જરૂરી છે.",
    "હોસ્પિટલમાં દર્દીને યોગ્ય ચિકિત્સા મળે છે.",
    "કલાકારે દીવાલ પર સુંદર ચિત્ર બનાવ્યું છે.",
    "જીવન જીવવા માટે ખોરાક અને પાણી મુખ્ય જરૂરિયાત છે.",
    "પોતાનું કામ સમયસર પૂરું કરવું એ આપણી જવાબદારી છે.",
    "સમાચારપત્રમાં નવી નોકરી માટે સુંદર જાહેરાત આવી છે.",
    "સાચું અને સુખી જીવન જીવવું એ કળા છે.",
    "પર્વત પરથી ઠંડા પાણીનું મીઠું ઝરણું વહે છે.",
    "ટપાલી ઘરે આવીને મહત્વની ટપાલ આપે છે.",
    "દવાઓ રાખવા માટે નાનો પ્લાસ્ટિકનો ડબ્બો વપરાય છે.",
    "આજના સમયમાં આધુનિક તકનીક નો ઉપયોગ વધ્યો છે.",
    "ઉત્તરાયણ અને હોળી એ ખુશીનો તહેવાર છે.",
    "દસ્તાવેજ પર આજની સાચી તારીખ લખેલી છે.",
    "જમીન અને મિલકત માટે સરકારી દસ્તાવેજ જરૂરી છે.",
    "મુસાફરે નકશામાં યોગ્ય દિશા જોઈ લીધી.",
    "દીવો પ્રગટાવવા માટે દીવાસળી ની પેટી વપરાય છે.",
    "વિદ્યાર્થી દસમા ધોરણ માં અભ્યાસ કરે છે.",
    "શાળામાં શિસ્ત જાળવવા માટે દરેક નિયમ પાળવો જોઈએ.",
    "વિચારપૂર્વક લીધેલો નિર્ણય સફળતા અપાવે છે.",
    "શહેરમાં નવા પુલ અને રસ્તાનું નિર્માણ થઈ રહ્યું છે.",
    "આપણી પ્રાચીન સંસ્કૃતિ અને પરંપરા મહાન છે.",
    "ગંભીર પરિસ્થિતિ માં ધીરજ રાખવી જોઈએ.",
    "આપણે વૃક્ષો વાવીને પર્યાવરણ નું રક્ષણ કરવું જોઈએ.",
    "સૂર્ય આપણને દિવસ દરમિયાન તેજસ્વી પ્રકાશ આપે છે.",
    "શિક્ષણ મેળવીને વ્યક્તિ જીવનમાં સારી પ્રગતિ કરે છે.",
    "રજાઓમાં પરિવાર સાથે પ્રવાસ જવાની મજા આવે છે.",
    "શાળામાં રમતગમતની વિવિધ પ્રવૃત્તિ થાય છે.",
    "નવા કર્મચારીઓને કામનું યોગ્ય પ્રશિક્ષણ આપવામાં આવે છે.",
    "સરકારે આ વર્ષ માટે નવું બજેટ રજૂ કર્યું.",
    "સમય સાથે સમાજમાં મોટો બદલાવ આવી રહ્યો છે.",
    "ઘરનું બારણું હંમેશા બંધ રાખવું જોઈએ.",
    "ગામડામાં વિતાવેલું બાળપણ ખૂબ સુંદર હોય છે.",
    "સારી મહેનત કરવાથી ઉજ્જવળ ભવિષ્ય બને છે.",
    "અમદાવાદ ગુજરાતનું સૌથી મોટું મહાનગર છે.",
    "સભામાં મહત્વનો મુદ્દો ચર્ચા માટે રજૂ થયો.",
    "ગઈકાલે જૂના મિત્ર સાથે આનંદદાયક મુલાકાત થઈ.",
    "યાત્રાળુઓ પવિત્ર મંદિરની યાત્રા કરવા નીકળ્યા.",
    "સૈનિકો સરહદ પર દેશનું રક્ષણ કરે છે.",
    "લેખકે નવી વાર્તાની સુંદર રચના કરી છે.",
    "સતત પરિશ્રમ કરવાથી મોટો લાભ થાય છે.",
    "ગામ અને શહેરનો સારો વિકાસ થઈ રહ્યો છે.",
    "ગાંધીજીની વિચારધારા અહિંસા અને સત્ય પર આધારિત છે.",
    "વિદ્યાર્થીએ શિક્ષકને રજા માટે વિનંતી કરી.",
    "સરકારી કચેરીમાં અલગ અલગ વિભાગ હોય છે.",
    "કોઈપણ સાચા મિત્ર સાથે વિશ્વાસઘાત કરવો ન જોઈએ.",
    "શાળા અને કોલેજમાં યોગ્ય શિસ્ત હોવી જરૂરી છે.",
    "મોબાઈલ ફોન દ્વારા મિત્રને સંદેશ મોકલ્યો.",
    "જરૂર પડે ત્યારે અધિકારી સાથે સંપર્ક કરવો.",
    "પરિવારના સભ્યો વચ્ચે પ્રેમભર્યો સંબંધ હોવો જોઈએ.",
    "બે નેતાઓ વચ્ચે શાંતિ માટે સંવાદ થયો.",
    "પોલીસ નાગરિકોની સુરક્ષા માટે કામ કરે છે.",
    "શહેરમાં મુસાફરી માટે ઉત્તમ સુવિધા ઉપલબ્ધ છે.",
    "આઝાદી પછી દેશને સંપૂર્ણ સ્વતંત્રતા મળી.",
    "દુકાનદાર પોતાના વેપારનો હિસાબ રાખે છે.",
    "મહેનત કરવાથી સારી આવક પ્રાપ્ત થાય છે.",
    "કચેરીમાં કામની યોગ્ય વ્યવસ્થા ગોઠવવામાં આવી છે.",

    # Everyday conversational sentences & common vocabulary (ઘરે, મિત્રો, સાથે, વાત, શાળા, ગરમી, ગરમ, etc.)
    "બાળક ઘરે મિત્રો સાથે વાત કરે છે.",
    "મારો મિત્ર દરરોજ શાળા જાય છે.",
    "વિદ્યાર્થીઓ શાળામાં ઉત્સાહથી ભણે છે.",
    "આજે ખૂબ ગરમી લાગે છે અને વાતાવરણ ગરમ છે.",
    "અમારા ગામ માં સુંદર નદી વહે છે.",
    "માતા ઘરે સ્વાદિષ્ટ ભોજન બનાવે છે.",
    "પિતા પુત્ર સાથે જરૂરી વાતચીત કરે છે.",
    "મિત્રો સાથે મળીને આનંદ કરે છે.",
    "દરેક માણસ પોતાના ઘરે સુખી રહે છે.",
    "તેણે પોતાના મન ની વાત કહી દીધી.",
    "વિદ્યાર્થી પુસ્તક વાંચીને જ્ઞાન મેળવે છે.",
    "ખેડૂત ખેતરમાં સખત મહેનત કરે છે.",
    "ડોક્ટર દર્દીની સારવાર કરે છે.",
    "આપણે સૌ સાથે મળીને કામ કરીએ છીએ.",
    "ગામમાં સારા માણસો વસે છે.",
    "શિક્ષક વિદ્યાર્થીને સારો પાઠ ભણાવે છે.",
    "બાળકો બગીચામાં રમત રમે છે.",
    "સૂર્ય પૂર્વ દિશામાં ઊગે છે.",
    "પાણી જીવન માટે અમૂલ્ય છે.",
    "પુસ્તક વાંચવું એ સારી આદત છે."
]


def train():
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass

    print("=" * 50)
    print("      TRAINING GUJARATI N-GRAM LANGUAGE MODEL     ")
    print("=" * 50)

    base_dir = Path(__file__).resolve().parent
    dataset_path = base_dir / "data" / "gujarati_training.csv"
    output_dir = base_dir / "model"
    output_path = output_dir / "ngram_model.json"

    if not dataset_path.exists():
        print(f"Error: Training dataset not found at {dataset_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/5] Loading training dataset from '{dataset_path}'...")
    sentences = []
    seen = set()
    with open(dataset_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = row.get("sentence", "").strip()
            if s and s not in seen:
                sentences.append(s)
                seen.add(s)

    # Incorporate authentic domain sentences to ensure vocabulary completeness
    added_count = 0
    for s in AUTHENTIC_CORPUS_SENTENCES:
        s_clean = s.strip()
        if s_clean and s_clean not in seen:
            sentences.append(s_clean)
            seen.add(s_clean)
            added_count += 1

    total_sentences = len(sentences)
    print(f"      Loaded {total_sentences} training sentences ({added_count} domain context sentences integrated).")

    print("\n[2/5] Preprocessing and tokenizing sentences...")
    tokenized_corpus = []
    for s in sentences:
        tokens = preprocess(s)
        if tokens:
            tokenized_corpus.append(tokens)

    print(f"      Successfully tokenized {len(tokenized_corpus)} sentences.")

    print("\n[3/5] Building vocabulary & counting N-grams (Unigrams, Bigrams, Trigrams)...")
    model = NGramLanguageModel()
    model.train(tokenized_corpus)

    print(f"      Total Words Counted:   {model.total_words}")
    print(f"      Vocabulary Size:       {model.vocab_size} unique words")
    print(f"      Distinct Bigrams:      {len(model.bigram_counts)}")
    print(f"      Distinct Trigrams:     {len(model.trigram_counts)}")

    print(f"\n[4/5] Serializing trained model to '{output_path}'...")
    model.save(str(output_path))
    print(f"      Model saved successfully ({output_path.stat().st_size} bytes).")

    print("\n[5/5] Sample Model Statistics:")
    print("      Top 5 Most Frequent Words (Unigrams):")
    for word, count in model.unigram_counts.most_common(5):
        prob = model.get_unigram_prob(word)
        print(f"        - {word}: count={count}, P(w)={prob:.4f}")

    print("\n" + "=" * 50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Trained model is available at: {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    train()
