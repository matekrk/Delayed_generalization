# NLP Datasets for Delayed Generalization Research

This directory contains natural language processing datasets that exhibit delayed generalization phenomena, particularly in learning semantic vs syntactic features and compositional understanding.

## 📊 Dataset Categories

### Compositional Understanding Tasks
- **Task**: Learning to combine known elements in novel ways
- **Phenomenon**: Initial memorization of seen combinations, delayed generalization to unseen compositions
- **Timeline**: 100-1000 epochs depending on complexity
- **Example**: Learning "red circle" and "blue square" then generalizing to "red square"

### Sentiment Analysis with Spurious Features
- **Task**: Sentiment classification with dataset bias
- **Phenomenon**: Learning surface patterns before deeper linguistic understanding  
- **Timeline**: 50-500 epochs for transition
- **Example**: Learning positive words vs understanding sentiment composition

### Natural Language Inference (NLI)
- **Task**: Determine logical relationship between premise and hypothesis
- **Phenomenon**: Learning lexical overlap heuristics before logical reasoning
- **Timeline**: 200-1000 epochs for robust reasoning
- **Example**: Moving from keyword matching to understanding entailment

### Language Modeling with Phase Transitions
- **Task**: Next token prediction with emergent abilities
- **Phenomenon**: Sudden improvement in specific capabilities (syntax, semantics)
- **Timeline**: Highly variable, depends on model size and data
- **Example**: Sudden emergence of grammatical understanding

## 🔬 Common NLP Delayed Generalization Patterns

### Surface vs Deep Feature Learning
1. **Surface Learning (Early)**: Model learns superficial patterns (word frequency, n-grams)
2. **Intermediate Phase**: Performance plateau on surface features
3. **Deep Learning (Late)**: Model develops semantic/syntactic understanding
4. **Robust Generalization**: Performance on out-of-distribution examples

### Compositional Generalization
1. **Memorization Phase**: Learning seen word combinations
2. **Limited Generalization**: Some transfer to similar compositions
3. **Systematic Generalization**: Understanding compositional rules
4. **Full Compositionality**: Novel combinations from learned elements

## 🛠️ Generating NLP Delayed Generalization Datasets

### Compositional Language Dataset
```python
def generate_compositional_dataset(
    adjectives=['red', 'blue', 'green', 'yellow'],
    nouns=['circle', 'square', 'triangle', 'star'],
    train_composition_ratio=0.7,
    test_novel_compositions=True
):
    """Generate dataset for studying compositional generalization"""
    
    # Generate all possible adjective-noun combinations
    all_combinations = []
    for adj in adjectives:
        for noun in nouns:
            combination = f"{adj} {noun}"
            # Task: predict color given shape or vice versa
            all_combinations.append({
                'text': combination,
                'adjective': adj,
                'noun': noun,
                'label': adjectives.index(adj)  # Color classification task
            })
    
    # Split: seen combinations vs novel combinations
    np.random.shuffle(all_combinations)
    split_point = int(len(all_combinations) * train_composition_ratio)
    
    train_combinations = all_combinations[:split_point]
    
    if test_novel_compositions:
        # Test on completely novel combinations
        test_combinations = all_combinations[split_point:]
    else:
        # Test on same distribution
        test_combinations = train_combinations[-len(all_combinations)//4:]
    
    return train_combinations, test_combinations

# Usage
train_data, test_data = generate_compositional_dataset()
```

### Biased Sentiment Dataset
```python
def generate_biased_sentiment_dataset(
    bias_strength=0.8,
    spurious_features=['amazing', 'terrible', 'great', 'awful'],
    neutral_content_variety=1000
):
    """Generate sentiment dataset with spurious word-sentiment correlations"""
    
    train_data = []
    test_data = []
    
    # Generate biased training data
    for sentiment in ['positive', 'negative']:
        for _ in range(1000):
            # Generate base content
            base_content = generate_neutral_content()
            
            if np.random.random() < bias_strength:
                # Add spurious feature correlated with sentiment
                if sentiment == 'positive':
                    spurious_word = np.random.choice(['amazing', 'great'])
                else:
                    spurious_word = np.random.choice(['terrible', 'awful'])
                
                content = f"{base_content} This is {spurious_word}."
            else:
                # Add opposite spurious feature (minority)
                if sentiment == 'positive':
                    spurious_word = np.random.choice(['terrible', 'awful'])
                else:
                    spurious_word = np.random.choice(['amazing', 'great'])
                
                content = f"{base_content} This is {spurious_word}."
            
            train_data.append({
                'text': content,
                'label': 1 if sentiment == 'positive' else 0,
                'spurious_word': spurious_word
            })
    
    # Generate unbiased test data (balanced spurious features)
    for sentiment in ['positive', 'negative']:
        for spurious_word in spurious_features:
            for _ in range(50):  # 50 examples per combination
                base_content = generate_neutral_content()
                content = f"{base_content} This is {spurious_word}."
                
                test_data.append({
                    'text': content,
                    'label': 1 if sentiment == 'positive' else 0,
                    'spurious_word': spurious_word
                })
    
    return train_data, test_data

def generate_neutral_content():
    """Generate neutral content without sentiment words"""
    templates = [
        "The product arrived on time and matched the description",
        "I ordered this item last week for my project",
        "The packaging was standard and the item was as expected",
        "This was purchased as a replacement for my old one"
    ]
    return np.random.choice(templates)
```

### Systematic NLI Dataset
```python
def generate_systematic_nli_dataset():
    """Generate NLI dataset testing systematic generalization"""
    
    # Templates for different logical relationships
    entailment_templates = [
        ("All {category} are {property}", "{item} is a {category}", "entailment"),
        ("The {object} is {color}", "There is a {color} {object}", "entailment"),
        ("{person} owns {number} {items}", "{person} owns at least one {item}", "entailment")
    ]
    
    contradiction_templates = [
        ("No {category} are {property}", "{item} is a {property} {category}", "contradiction"),
        ("The {object} is {color1}", "The {object} is {color2}", "contradiction"),
        ("{person} owns zero {items}", "{person} owns {number} {items}", "contradiction")
    ]
    
    neutral_templates = [
        ("Some {category} are {property}", "{item} is a {category}", "neutral"),
        ("The {object} exists", "The {object} is {color}", "neutral"),
        ("{person} likes {activity}", "{person} owns {items}", "neutral")
    ]
    
    # Generate training data with limited vocabulary
    train_vocab = {
        'category': ['animals', 'vehicles'],
        'property': ['fast', 'large'],
        'item': ['dog', 'car'],
        'object': ['ball', 'box'],
        'color': ['red', 'blue'],
        'color1': ['red'], 'color2': ['blue'],
        'person': ['John', 'Mary'],
        'number': ['two', 'three'],
        'items': ['books', 'pens'],
        'item': ['book', 'pen'],
        'activity': ['reading', 'writing']
    }
    
    # Generate test data with expanded vocabulary (systematic generalization)
    test_vocab = train_vocab.copy()
    test_vocab.update({
        'category': train_vocab['category'] + ['furniture', 'tools'],
        'property': train_vocab['property'] + ['small', 'slow'],
        'item': train_vocab['item'] + ['chair', 'hammer'],
        'person': train_vocab['person'] + ['Alice', 'Bob']
    })
    
    def generate_examples(templates, vocab, num_examples=1000):
        examples = []
        for _ in range(num_examples):
            template = np.random.choice(templates)
            premise_template, hypothesis_template, label = template
            
            # Fill in template with vocabulary
            filled_vars = {}
            for var in vocab:
                if f"{{{var}}}" in premise_template or f"{{{var}}}" in hypothesis_template:
                    filled_vars[var] = np.random.choice(vocab[var])
            
            premise = premise_template.format(**filled_vars)
            hypothesis = hypothesis_template.format(**filled_vars)
            
            examples.append({
                'premise': premise,
                'hypothesis': hypothesis,
                'label': label
            })
        
        return examples
    
    # Generate training and test sets
    all_templates = entailment_templates + contradiction_templates + neutral_templates
    
    train_data = generate_examples(all_templates, train_vocab, 3000)
    test_data = generate_examples(all_templates, test_vocab, 1000)
    
    return train_data, test_data
```

## 📈 Training Protocols for NLP Delayed Generalization

### Transformer Training for Compositional Tasks
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def setup_compositional_training():
    """Setup for compositional generalization experiments"""
    
    # Use smaller model for controlled experiments
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=4  # Number of colors/adjectives
    )
    
    config = {
        "epochs": 500,
        "batch_size": 32,
        "learning_rate": 2e-5,
        "optimizer": "AdamW",
        "weight_decay": 1e-2,
        "warmup_steps": 100,
        
        # Compositional-specific monitoring
        "eval_interval": 10,
        "track_compositional_accuracy": True,
        "save_attention_maps": True
    }
    
    return model, tokenizer, config

def track_compositional_learning(model, tokenizer, train_data, test_data, epoch):
    """Track progress on compositional understanding"""
    
    # Evaluate on seen compositions
    seen_acc = evaluate_compositions(model, tokenizer, train_data)
    
    # Evaluate on novel compositions  
    novel_acc = evaluate_compositions(model, tokenizer, test_data)
    
    # Analyze attention patterns
    attention_analysis = analyze_compositional_attention(
        model, tokenizer, test_data
    )
    
    return {
        "epoch": epoch,
        "seen_composition_accuracy": seen_acc,
        "novel_composition_accuracy": novel_acc,
        "compositional_gap": seen_acc - novel_acc,
        "attention_analysis": attention_analysis
    }
```

### Bias Mitigation in NLP
```python
def setup_bias_mitigation_training():
    """Setup for studying and mitigating linguistic bias"""
    
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2  # Binary sentiment
    )
    
    # Bias mitigation techniques
    training_strategies = {
        "standard": {
            "method": "erm",
            "data_augmentation": False
        },
        "reweighting": {
            "method": "reweight_by_group",
            "group_weights": calculate_group_weights(train_data)
        },
        "adversarial": {
            "method": "adversarial_debiasing",
            "adversarial_lambda": 0.1
        },
        "data_augmentation": {
            "method": "erm",
            "data_augmentation": True,
            "augmentation_type": "spurious_word_replacement"
        }
    }
    
    return model, training_strategies

def evaluate_bias_mitigation(model, tokenizer, test_data):
    """Evaluate how well bias mitigation worked"""
    
    # Overall accuracy
    overall_acc = evaluate_standard_accuracy(model, tokenizer, test_data)
    
    # Group-wise accuracy (by spurious feature)
    group_accuracies = {}
    for spurious_word in ['amazing', 'terrible', 'great', 'awful']:
        group_data = [ex for ex in test_data if ex['spurious_word'] == spurious_word]
        group_acc = evaluate_standard_accuracy(model, tokenizer, group_data)
        group_accuracies[spurious_word] = group_acc
    
    # Spurious correlation measure
    spurious_reliance = measure_spurious_reliance(model, tokenizer, test_data)
    
    return {
        "overall_accuracy": overall_acc,
        "group_accuracies": group_accuracies,
        "worst_group_accuracy": min(group_accuracies.values()),
        "spurious_reliance": spurious_reliance
    }

def measure_spurious_reliance(model, tokenizer, test_data):
    """Measure how much model relies on spurious features"""
    
    # Original performance
    original_acc = evaluate_standard_accuracy(model, tokenizer, test_data)
    
    # Performance with spurious words masked
    masked_data = []
    for example in test_data:
        masked_text = example['text'].replace(example['spurious_word'], '[MASK]')
        masked_data.append({
            'text': masked_text,
            'label': example['label']
        })
    
    masked_acc = evaluate_standard_accuracy(model, tokenizer, masked_data)
    
    # Spurious reliance = performance drop when spurious features removed
    spurious_reliance = (original_acc - masked_acc) / original_acc
    
    return spurious_reliance
```

## 📊 Evaluation Metrics for NLP Delayed Generalization

### Compositional Generalization Metrics
```python
def evaluate_compositional_generalization(model, tokenizer, data):
    """Comprehensive evaluation of compositional understanding"""
    
    metrics = {}
    
    # 1. Seen vs Novel Composition Accuracy
    seen_compositions = get_seen_compositions(data)
    novel_compositions = get_novel_compositions(data)
    
    metrics['seen_accuracy'] = evaluate_compositions(model, tokenizer, seen_compositions)
    metrics['novel_accuracy'] = evaluate_compositions(model, tokenizer, novel_compositions)
    metrics['compositional_gap'] = metrics['seen_accuracy'] - metrics['novel_accuracy']
    
    # 2. Systematic Generalization by Type
    for comp_type in ['adjective_noun', 'verb_object', 'quantifier_noun']:
        type_data = filter_by_composition_type(data, comp_type)
        metrics[f'{comp_type}_accuracy'] = evaluate_compositions(model, tokenizer, type_data)
    
    # 3. Productivity (novel element combinations)
    productivity_score = evaluate_productivity(model, tokenizer, data)
    metrics['productivity'] = productivity_score
    
    # 4. Substitutivity (replacing elements in compositions)
    substitutivity_score = evaluate_substitutivity(model, tokenizer, data)
    metrics['substitutivity'] = substitutivity_score
    
    return metrics

def evaluate_productivity(model, tokenizer, data):
    """Measure ability to use known elements in novel positions"""
    
    # Test: If model knows "red circle" and "blue square", 
    # can it handle "red square" and "blue circle"?
    
    known_pairs = extract_known_pairs(data)
    novel_combinations = generate_novel_combinations(known_pairs)
    
    accuracy = evaluate_compositions(model, tokenizer, novel_combinations)
    return accuracy

def evaluate_substitutivity(model, tokenizer, data):
    """Measure ability to substitute similar elements"""
    
    # Test: If "big dog" and "large cat" are known,
    # can model handle "big cat" and "large dog"?
    
    substitution_pairs = find_substitution_pairs(data)
    substituted_examples = apply_substitutions(substitution_pairs)
    
    accuracy = evaluate_compositions(model, tokenizer, substituted_examples)
    return accuracy
```

### Temporal Analysis of Learning
```python
def analyze_learning_progression(training_logs):
    """Analyze how different capabilities develop over time"""
    
    capabilities = {
        'surface_features': [],
        'syntactic_understanding': [],
        'semantic_understanding': [],
        'compositional_reasoning': []
    }
    
    for epoch_log in training_logs:
        # Surface features (word frequency, n-grams)
        surface_score = evaluate_surface_features(epoch_log)
        capabilities['surface_features'].append(surface_score)
        
        # Syntactic understanding (grammar, POS)
        syntax_score = evaluate_syntactic_understanding(epoch_log)
        capabilities['syntactic_understanding'].append(syntax_score)
        
        # Semantic understanding (word meanings, relations)
        semantic_score = evaluate_semantic_understanding(epoch_log)
        capabilities['semantic_understanding'].append(semantic_score)
        
        # Compositional reasoning (novel combinations)
        compositional_score = evaluate_compositional_reasoning(epoch_log)
        capabilities['compositional_reasoning'].append(compositional_score)
    
    # Detect phase transitions for each capability
    transitions = {}
    for capability, scores in capabilities.items():
        transition_epoch = detect_phase_transition(scores)
        transitions[capability] = transition_epoch
    
    return capabilities, transitions
```

## 🎯 Expected Results and Timelines

### Compositional Generalization
- **Phase 1 (0-100 epochs)**: Memorization of seen combinations
- **Phase 2 (100-300 epochs)**: Limited transfer to similar compositions  
- **Phase 3 (300-800 epochs)**: Systematic compositional understanding
- **Performance**: 60-80% novel composition accuracy after transition

### Sentiment Bias Mitigation
- **Phase 1 (0-50 epochs)**: Learning spurious word-sentiment correlations
- **Phase 2 (50-200 epochs)**: High accuracy on correlated examples
- **Phase 3 (200-500 epochs)**: Gradual learning of semantic sentiment
- **Performance**: <20% performance gap between biased and unbiased test sets

### NLI Systematic Generalization
- **Phase 1 (0-100 epochs)**: Learning lexical overlap heuristics
- **Phase 2 (100-400 epochs)**: Improving on training distribution
- **Phase 3 (400-1000 epochs)**: True logical reasoning emergence
- **Performance**: >80% accuracy on systematic generalization test

## 🔗 Integration with Existing Frameworks

### Hugging Face Integration
```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

def create_hf_dataset(data, tokenizer, max_length=128):
    """Convert to Hugging Face Dataset format"""
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            truncation=True, 
            padding='max_length',
            max_length=max_length
        )
    
    dataset = Dataset.from_list(data)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    return tokenized_dataset

# Usage with delayed generalization monitoring
def train_with_delayed_generalization_monitoring(
    model, train_dataset, eval_dataset, output_dir
):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=500,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=50,
        save_steps=500,
        learning_rate=2e-5,
        weight_decay=1e-2,
        warmup_steps=100,
    )
    
    trainer = DelayedGeneralizationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    trainer.train()
    return trainer
```

## 📚 References

1. Lake & Baroni (2018). "Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks"
2. Sagawa et al. (2020). "Distributionally Robust Neural Networks for Group Shifts"
3. McCoy et al. (2019). "Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference"
4. Tenney et al. (2019). "BERT Rediscovers the Classical NLP Pipeline"
5. Rogers et al. (2020). "A Primer on Neural Network Models for Natural Language Processing"