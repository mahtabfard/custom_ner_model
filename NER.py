import spacy 
from spacy.training.example import Example
import random
from data import get_training_data
#loading small model 
nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "attribute_ruler", "lemmatizer"])
#getting ner pipeline + adding new labels
ner = nlp.get_pipe("ner")
ner.add_label("JUDGE")
ner.add_label("LOCATION")
ner.add_label("PERSON")
ner.add_label("ORG")
#disabaling pther components in pipeline to onlu use and train ner
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
n_iter = 50
#training
TRAINING_DA = get_training_data()
with nlp.disable_pipes(*unaffected_pipes):
    optimizer = nlp.begin_training()
    for i in range(n_iter):
        random.shuffle(TRAINING_DA)
        losses={}
        for text , annotations in TRAINING_DA:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], losses=losses, drop=0.5, sgd=optimizer)
        print(f"Iteration {i +1} , losses {losses}")
# Test the model
test_text = "Judge Brown ruled in the case located in Paris."
doc = nlp(test_text)
for ent in doc.ents:
    print(ent.text, ent.label_)

# Save the model
nlp.to_disk("custom_ner_model")