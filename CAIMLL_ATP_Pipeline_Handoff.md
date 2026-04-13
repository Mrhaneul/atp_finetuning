# CAIMLL ATP 2-01.3 Pipeline — Implementation Handoff
## Everything needed to build the pipeline code

**Companion to:** `CAIMLL_ATP_Strategy_FINAL.md`  
**Purpose:** This file contains every constant, schema, template, file path, and implementation detail needed to write the pipeline scripts. A developer or Claude Code session should be able to read this file + the strategy file and produce working code without further clarification.

---

## SECTION 1: ENVIRONMENT & INFRASTRUCTURE

### Hardware
- **Platform:** NVIDIA DGX Spark GB10 Grace Blackwell Superchip
- **VRAM:** ~119GB total
- **Hostname:** spark-87dc (shared environment)
- **CUDA:** 13.0 (GB10 kernel)
- **PyTorch:** 2.11.0+cu128
- **Note:** float16 causes infinity on Gemma — must use bfloat16

### Software Stack
```
Python 3.11+
torch==2.11.0+cu128  (--index-url https://download.pytorch.org/whl/cu128)
unsloth              (latest — Gemma 4 support confirmed)
transformers>=4.51.0 (Gemma 4 support)
peft
trl
bitsandbytes
datasets
ollama               (port 11434, already installed)
```

### Key Infrastructure Lessons (from FM 2-0 pipeline)
- `pdftotext` without `-layout` flag is the correct extraction method on DGX
- Ollama must be stopped before training: `ollama stop gemma4:31b`
- Verify VRAM free: `nvidia-smi`
- Seeds file must use append mode (`"a"`) not write mode (`"w"`) for resumability
- Always use `coerce_str()` to handle Ollama returning output as dict/list
- Background execution: `nohup python run_pipeline.py > pipeline.log 2>&1 &`
- Monitor with tmux split panes (pipeline left, monitor.py right)

### File Paths
```
PROJECT ROOT:       ~/caimll_finetuning/atp_pipeline_v2/
SOURCE PDF:         /mnt/project/ATP_2-01_3_Intelligence_Preparation_of_the_Battlefield.pdf
FM 2-0 REFERENCE:   /mnt/project/FM2-0.pdf
STRATEGY DOC:       /mnt/project/CAIMLL_ATP_Strategy_FINAL.md  (or wherever saved)
PREVIOUS PIPELINE:  ~/caimll_finetuning/pipeline_v2/  (FM 2-0 V2b — reference implementation)
BEST FM2 ADAPTER:   ~/caimll_finetuning/pipeline_v2/outputs/fm2-llama3-lora-v2b/
```

---

## SECTION 2: MODEL CONFIGURATION

### Generator Model (Stages 1-3)
```
Model:          gemma4:31b  (via Ollama)
Ollama pull:    ollama pull gemma4:31b
API endpoint:   http://localhost:11434/api/generate
Thinking mode:  Enabled via system prompt containing <|think|>
Context window: 256K tokens
Chat template:  Standard system/assistant/user roles (not Gemma 3 style)
```

### Ollama API call pattern
```python
import requests
import json

def generate(prompt: str, system: str = "", temperature: float = 0.7) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma4:31b",
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_k": 64,
                "top_p": 0.95,
                "num_predict": 4096
            }
        },
        timeout=300
    )
    result = response.json()
    return coerce_str(result.get("response", ""))

def coerce_str(val) -> str:
    """Handle Ollama returning output as dict/list/other types."""
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return json.dumps(val)
    if isinstance(val, list):
        return " ".join(str(v) for v in val)
    return str(val)
```

### Fine-Tune Target Model (Stage 4)
```
Model:          google/gemma-4-31B-it  (or unsloth/gemma-4-31B-it-unsloth-bnb-4bit if available)
Framework:      Unsloth + QLoRA
Quantization:   4-bit NF4 (bitsandbytes)
VRAM estimate:  ~22GB with QLoRA
Max seq length: 4096
```

### Training Hyperparameters
```python
TRAIN_CONFIG = {
    # LoRA
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", 
                            "gate_proj", "up_proj", "down_proj"],
    "use_rslora": True,
    
    # Training
    "num_train_epochs": 2,        # Start with 2; increase to 3 if loss > 1.3
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,  # Effective batch = 16
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "optim": "adamw_8bit",
    "bf16": True,                 # MUST be bf16, not fp16 (Gemma 4 overflows fp16)
    "max_seq_length": 4096,
    "logging_steps": 10,
    "save_strategy": "epoch",
    
    # Loss targets
    "target_loss_min": 0.9,       # Below this = overfitting risk
    "target_loss_max": 1.3,       # Above this = underfitting risk
    "target_loss_sweet": 1.05,    # Ideal
}
```

---

## SECTION 3: ATP 2-01.3 DOCUMENT STRUCTURE

### Chapter Map (critical for chunker and metadata validation)
```python
ATP_CHAPTERS = {
    1: {
        "title": "Intelligence Preparation of the Battlefield Fundamentals",
        "ipb_step": 0,  # Fundamentals, not a specific step
        "weight": 2.5,
        "description": "IPB definition, operational framework, peer threats, multi-domain ops, windows of opportunity"
    },
    2: {
        "title": "IPB Support to Planning and Decision Making",
        "ipb_step": 0,  # Support/integration, not a step
        "weight": 2.0,
        "description": "IPB-MDMP relationship, troop leading procedures, integrating processes"
    },
    3: {
        "title": "Step 1—Define the Operational Environment",
        "ipb_step": 1,
        "weight": 2.0,
        "description": "AO/AI limits, significant characteristics, evaluate holdings, initiate collection"
    },
    4: {
        "title": "Step 2—Describe Environmental Effects on Operations",
        "ipb_step": 2,
        "weight": 2.0,
        "description": "Threat effects, terrain (OAKOC), weather effects, civil considerations, MCOO"
    },
    5: {
        "title": "Step 3—Evaluate the Threat",
        "ipb_step": 3,
        "weight": 2.5,
        "description": "Threat characteristics, threat models, threat capabilities, doctrinal templates"
    },
    6: {
        "title": "Step 4—Determine Threat Courses of Action",
        "ipb_step": 4,
        "weight": 2.5,
        "description": "Threat COA development, situation templates, event templates, HVT lists"
    },
    7: {
        "title": "IPB Support to Offense, Defense, Stability, and Unique Environments",
        "ipb_step": None,  # Cross-cutting
        "weight": 3.0,
        "description": "Offense/defense/stability IPB + urban, littoral, subterranean environments"
    },
    8: {
        "title": "IPB Across Domains",
        "ipb_step": None,  # Cross-cutting
        "weight": 2.0,
        "description": "Air, maritime, space, cyberspace, information environment, EMS"
    },
    "A": {
        "title": "Intelligence Staff Officer IPB Checklist",
        "ipb_step": None,
        "weight": 1.5,
        "description": "Procedural checklist for S-2 conducting IPB"
    },
    "B": {
        "title": "Tools for Use During IPB",
        "ipb_step": None,
        "weight": 1.5,
        "description": "Analytical tools, matrices, overlays"
    },
    "C": {
        "title": "Threat Characteristics for Regular, Irregular, and Hybrid Threats",
        "ipb_step": 3,  # Directly supports Step 3
        "weight": 2.0,
        "description": "Detailed threat type characteristics"
    },
    "D": {
        "title": "IPB Cyberspace Considerations",
        "ipb_step": None,
        "weight": 1.5,
        "description": "Cyber Kill Chain, cyberspace integration into each IPB step"
    }
}
```

### Paragraph ID Pattern
```python
import re
PARA_PATTERN = re.compile(r'^(\d+-\d+)\.\s')
# Matches: "1-1. ", "3-24. ", "6-15. ", etc.
# Appendix pattern:
APPENDIX_PARA_PATTERN = re.compile(r'^([A-D]-\d+)\.\s')
# Matches: "A-1. ", "C-12. ", "D-33. ", etc.
```

---

## SECTION 4: DATA SCHEMAS

### chunks.jsonl — Stage 1 output
```json
{
  "chunk_id": "atp201_3-ch3-p24",
  "para_id": "3-24",
  "text": "Full paragraph text...",
  "chapter_num": 3,
  "chapter_title": "Step 1—Define the Operational Environment",
  "section": "Identify Significant Characteristics",
  "page_range": [42, 43],
  "word_count": 187,
  "has_figure_reference": false,
  "cross_references": ["para 1-4", "FM 2-0"]
}
```

### enriched.jsonl — Stage 2 output (chunks + grounded metadata)
```json
{
  "chunk_id": "atp201_3-ch3-p24",
  "para_id": "3-24",
  "text": "Full paragraph text...",
  "chapter_num": 3,
  "chapter_title": "Step 1—Define the Operational Environment",
  "section": "Identify Significant Characteristics",
  "page_range": [42, 43],
  "word_count": 187,
  "has_figure_reference": false,
  "cross_references": ["para 1-4", "FM 2-0"],
  "metadata": {
    "ipb_step": 1,
    "content_type": "process",
    "echelon": "tactical",
    "domain": "land",
    "threat_type": null,
    "ipb_product": null,
    "environment": "general",
    "doctrinal_weight": 2.0
  }
}
```

### seeds.jsonl — Stage 3 output (QA triples with thinking traces)
```json
{
  "qa_id": "atp201_qa_0042",
  "source_chunks": ["atp201_3-ch3-p24", "atp201_3-ch3-p25"],
  "question_type": "applied_reasoning",
  "question": "A BCT S-2 is conducting IPB for an operation in a densely urbanized area...",
  "thinking_trace": "Let me work through this using ATP 2-01.3 doctrine.\n\n1. Step 2 of IPB focuses on...",
  "answer": "When conducting IPB Step 2 in urban terrain...\n\n[Reference: ATP 2-01.3, Ch. 4, Ch. 7, para 4-12 through 4-30, para 7-25 through 7-35]",
  "metadata": {
    "ipb_step": 2,
    "content_type": "applied_reasoning",
    "echelon": "tactical",
    "domain": "land",
    "environment": "urban",
    "difficulty": "hard",
    "citation_paragraphs": ["4-12", "4-30", "7-25", "7-35"]
  }
}
```

### train.jsonl — Stage 4 output (Gemma 4 chat template formatted)
```json
{
  "text": "<bos><|system|>\n<|think|>\nYou are a military intelligence analyst with deep expertise in ATP 2-01.3, Intelligence Preparation of the Battlefield. This question concerns Step 2—Describe Environmental Effects on Operations (IPB Step 2), Chapter 4: Step 2—Describe Environmental Effects on Operations.\nEchelon context: tactical. Environment: urban.\n\nAlways reason step by step through the relevant doctrine before answering.\nCite specific ATP 2-01.3 paragraphs in your reasoning. Place reference citations at the end of your answer.\n<|end|>\n<|user|>\nA BCT S-2 is conducting IPB for an operation in a densely urbanized area...\n<|end|>\n<|assistant|>\n<|channel>thought\nLet me work through this using ATP 2-01.3 doctrine...\n<channel|>\nWhen conducting IPB Step 2 in urban terrain...\n\n[Reference: ATP 2-01.3, Ch. 4, Ch. 7, para 4-12 through 4-30]\n<|end|>"
}
```

---

## SECTION 5: PROMPT TEMPLATES

### Stage 2 — Metadata Classification Prompt
```python
ENRICHMENT_SYSTEM = """<|think|>
You are a military intelligence doctrine classifier. Given a paragraph from 
ATP 2-01.3 (Intelligence Preparation of the Battlefield), classify it with 
structural metadata. Every classification must be verifiable by reading the 
paragraph itself. Do NOT invent or add content beyond what the paragraph states.
Respond ONLY in valid JSON with no markdown formatting."""

ENRICHMENT_USER = """PARAGRAPH:
{chunk_text}

CHAPTER: {chapter_num} — {chapter_title}
PARAGRAPH ID: {para_id}

Classify this paragraph. All fields must be verifiable from the text:
{{
  "ipb_step": <0|1|2|3|4|null>,
  "content_type": <"definition"|"process"|"product"|"example"|"checklist"|"considerations"|"cross_reference">,
  "echelon": <"tactical"|"operational"|"strategic"|"multi">,
  "domain": <"land"|"air"|"maritime"|"space"|"cyberspace"|"information"|"multi_domain">,
  "threat_type": <"regular"|"irregular"|"hybrid"|"peer"|null>,
  "ipb_product": <"MCOO"|"sittemp"|"event_template"|"HVT_list"|"terrain_effects_matrix"|"weather_effects"|"civil_overlay"|null>,
  "environment": <"general"|"urban"|"littoral"|"subterranean"|null>,
  "doctrinal_weight": <1.0-3.0>
}}"""
```

### Stage 3 — QA Generation Prompt (per question type)
```python
GENERATION_SYSTEM = """<|think|>
You are a senior military intelligence analyst and doctrine instructor generating 
training data for a reasoning AI. You will generate a question, a detailed 
thinking trace, and a final answer based on the provided ATP 2-01.3 content.

RULES FOR THINKING TRACE:
- Start with "Let me work through this using ATP 2-01.3 doctrine."
- Reference specific paragraph numbers (e.g., "para 3-24", "para 6-15")
- Show multi-step doctrinal logic — do not just restate the source text
- For applied_reasoning: extend BEYOND the literal text to show how a trained 
  analyst would apply these principles to a scenario
- For contrastive: state what is correct AND why alternatives are wrong
- End with a synthesis before the final answer
- Maximum 500 tokens

RULES FOR ANSWER:
- Write in natural analyst voice — do NOT start with "According to ATP 2-01.3..."
- Be self-contained (readable without the thinking trace)
- Use proper military terminology
- End with a [Reference: ATP 2-01.3, ...] citation block
- For product_generation: describe the product structure, not just name it

RULES FOR QUESTION:
- Must be specific enough that only ATP 2-01.3 doctrine can fully answer it
- For applied_reasoning: include a concrete operational scenario
- Never answerable with a single sentence

Respond ONLY in valid JSON with no markdown formatting."""

GENERATION_USER = """SOURCE CONTENT:
{chunk_text}

METADATA:
- Chapter: {chapter_num} — {chapter_title}
- IPB Step: {ipb_step}
- Paragraph(s): {para_ids}
- Content Type: {content_type}
- Environment: {environment}
- Echelon: {echelon}

QUESTION TYPE TO GENERATE: {question_type}

Generate in this exact JSON format:
{{
  "question": "...",
  "thinking_trace": "...",
  "answer": "...",
  "difficulty": "easy|medium|hard",
  "citation_paragraphs": ["3-24", "3-25"]
}}"""
```

### Question Type Definitions (for routing in generator)
```python
QUESTION_TYPES = {
    "factual": {
        "description": "Direct recall of definitions, terms, lists",
        "think_depth": "short",
        "target_pct": 0.15,
        "instructions": "Ask for a specific definition, enumeration, or doctrinal fact. The thinking trace should identify the exact paragraph and quote the key terms."
    },
    "procedural": {
        "description": "Step-by-step process execution",
        "think_depth": "medium",
        "target_pct": 0.15,
        "instructions": "Ask how to execute a specific IPB procedure. The thinking trace should walk through each step referencing paragraph numbers."
    },
    "comparative": {
        "description": "Compare/contrast concepts, products, steps",
        "think_depth": "medium",
        "target_pct": 0.10,
        "instructions": "Ask for differences or similarities between two doctrinal concepts. The thinking trace should identify the relevant paragraphs for each concept and explicitly contrast them."
    },
    "echelon_specific": {
        "description": "Role differences across BCT/Division/Corps",
        "think_depth": "medium",
        "target_pct": 0.10,
        "instructions": "Ask how a task or responsibility differs between echelons. The thinking trace must distinguish specific echelon roles — never give a generic answer."
    },
    "applied_reasoning": {
        "description": "Apply doctrine to a novel scenario",
        "think_depth": "deep",
        "target_pct": 0.20,
        "instructions": "Provide a concrete operational scenario and ask how doctrine applies. The thinking trace must bridge from doctrinal definitions to scenario-specific application. This is where the model learns to reason BEYOND the manual text."
    },
    "product_generation": {
        "description": "Create/describe an IPB product",
        "think_depth": "deep",
        "target_pct": 0.10,
        "instructions": "Ask the model to describe or create a specific IPB product (MCOO, sit template, HVT list, event template). The thinking trace should reference the doctrinal product requirements."
    },
    "cross_step": {
        "description": "How output of one IPB step feeds another",
        "think_depth": "deep",
        "target_pct": 0.10,
        "instructions": "Ask how outputs from one IPB step serve as inputs to another. The thinking trace must trace the information flow between steps."
    },
    "multi_domain": {
        "description": "IPB across land/air/cyber/space/info domains",
        "think_depth": "deep",
        "target_pct": 0.05,
        "instructions": "Ask about IPB considerations specific to non-land domains. Reference Chapter 8 and Appendix D content."
    },
    "contrastive": {
        "description": "Why X and not Y — forces precise distinction",
        "think_depth": "medium",
        "target_pct": 0.05,
        "instructions": "Ask why one approach/product/concept is used instead of a similar alternative. The thinking trace must explain both what IS correct and why the alternative is NOT."
    }
}
```

---

## SECTION 6: QUALITY FILTERS

### Stage 2 — Metadata Validation
```python
def validate_metadata(chunk: dict) -> bool:
    """Validate that LLM-classified metadata is consistent with known chapter structure."""
    meta = chunk["metadata"]
    ch = chunk["chapter_num"]
    
    # Hard rules: chapter-to-ipb_step mapping
    CHAPTER_STEP_MAP = {3: 1, 4: 2, 5: 3, 6: 4}
    if ch in CHAPTER_STEP_MAP:
        if meta["ipb_step"] != CHAPTER_STEP_MAP[ch]:
            return False  # REJECT: ipb_step conflicts with chapter
    
    # content_type must be a valid enum
    VALID_TYPES = {"definition", "process", "product", "example", 
                   "checklist", "considerations", "cross_reference"}
    if meta["content_type"] not in VALID_TYPES:
        return False
    
    # doctrinal_weight in range
    if not (1.0 <= meta["doctrinal_weight"] <= 3.0):
        return False
    
    return True
```

### Stage 3 — QA Quality Filter
```python
def validate_qa(qa: dict) -> bool:
    """Filter out low-quality generated QA triples."""
    # Question must be >20 words
    if len(qa["question"].split()) < 20:
        return False
    
    # Answer must be >50 words
    if len(qa["answer"].split()) < 50:
        return False
    
    # Thinking trace must be >30 words
    if len(qa["thinking_trace"].split()) < 30:
        return False
    
    # Answer must end with [Reference: ...] block
    if "[Reference:" not in qa["answer"]:
        return False
    
    # Answer must NOT start with "According to ATP"
    if qa["answer"].strip().startswith("According to"):
        return False
    
    # Thinking trace must reference at least one paragraph number
    para_ref = re.search(r'para\s+\d+-\d+', qa["thinking_trace"])
    if not para_ref:
        return False
    
    # Thinking trace must not exceed 500 tokens (~375 words)
    if len(qa["thinking_trace"].split()) > 400:
        return False
    
    # citation_paragraphs must contain valid ATP para IDs
    for p in qa.get("citation_paragraphs", []):
        if not re.match(r'^[A-D]?-?\d+-?\d+$', p):
            return False
    
    return True
```

### Stage 3 — JSON Extraction from LLM Output
```python
def strip_think(text: str) -> str:
    """Remove <|channel>thought....<channel|> blocks from Gemma 4 output."""
    # Gemma 4 thinking format
    text = re.sub(r'<\|channel>thought\n.*?<channel\|>', '', text, flags=re.DOTALL)
    # Also handle any <think>...</think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()

def extract_json(text: str) -> dict:
    """Extract JSON from LLM output, handling markdown fences and thinking blocks."""
    text = strip_think(text)
    # Remove markdown code fences
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    return json.loads(text)
```

---

## SECTION 7: GEMMA 4 CHAT TEMPLATE FORMATTING

### System prompt template (injected per training example)
```python
SYSTEM_TEMPLATE = """<|think|>
You are a military intelligence analyst with deep expertise in ATP 2-01.3, \
Intelligence Preparation of the Battlefield. This question concerns \
{step_description}, Chapter {chapter_num}: {chapter_title}.
Echelon context: {echelon}. Environment: {environment}.

Always reason step by step through the relevant doctrine before answering.
Cite specific ATP 2-01.3 paragraphs in your reasoning. Place reference \
citations at the end of your answer."""

def format_system(meta: dict) -> str:
    step_desc = {
        0: "IPB Fundamentals",
        1: "Step 1—Define the Operational Environment (IPB Step 1)",
        2: "Step 2—Describe Environmental Effects on Operations (IPB Step 2)",
        3: "Step 3—Evaluate the Threat (IPB Step 3)",
        4: "Step 4—Determine Threat Courses of Action (IPB Step 4)",
        None: "cross-cutting IPB topics"
    }
    return SYSTEM_TEMPLATE.format(
        step_description=step_desc.get(meta.get("ipb_step"), "IPB topics"),
        chapter_num=meta.get("chapter_num", ""),
        chapter_title=meta.get("chapter_title", ""),
        echelon=meta.get("echelon", "general"),
        environment=meta.get("environment", "general")
    )
```

### Full training example formatter
```python
def format_training_example(qa: dict, chunk_meta: dict) -> str:
    """Format a QA triple into Gemma 4 chat template for SFT."""
    system = format_system(chunk_meta)
    
    return (
        f"<bos><|system|>\n{system}\n<|end|>\n"
        f"<|user|>\n{qa['question']}\n<|end|>\n"
        f"<|assistant|>\n"
        f"<|channel>thought\n{qa['thinking_trace']}\n<channel|>\n"
        f"{qa['answer']}\n<|end|>"
    )
```

---

## SECTION 8: CHAPTER-WEIGHTED SAMPLING

```python
CHAPTER_WEIGHTS = {
    1: 2.5, 2: 2.0, 3: 2.0, 4: 2.0,
    5: 2.5, 6: 2.5, 7: 3.0, 8: 2.0,
    "A": 1.5, "B": 1.5, "C": 2.0, "D": 1.5
}

def compute_questions_per_chunk(chunk: dict, total_target: int, 
                                 all_chunks: list) -> int:
    """Compute how many questions to generate per chunk based on chapter weight."""
    ch = chunk["chapter_num"]
    weight = CHAPTER_WEIGHTS.get(ch, 1.0)
    
    total_weight = sum(
        CHAPTER_WEIGHTS.get(c["chapter_num"], 1.0) for c in all_chunks
    )
    
    share = weight / total_weight
    raw_count = int(total_target * share / len([
        c for c in all_chunks if c["chapter_num"] == ch
    ]))
    
    return max(3, min(raw_count, 12))  # Clamp between 3 and 12 per chunk
```

---

## SECTION 9: EVALUATION SUITE

### 24 Eval Questions (questions.json)
```python
EVAL_QUESTIONS = [
    {"id": "F-01", "type": "factual", "ipb_step": 0,
     "question": "What is the definition of Intelligence Preparation of the Battlefield according to ATP 2-01.3?",
     "keywords": ["systematic", "process", "analyzing", "mission variables", "enemy", "terrain", "weather", "civil considerations", "area of interest", "effect on operations"]},
    
    {"id": "F-02", "type": "factual", "ipb_step": 0,
     "question": "What are the five categories of IPB products listed in ATP 2-01.3?",
     "keywords": ["situation templates", "COA statements", "HVT lists", "event templates", "event matrices", "MCOO", "terrain effects", "weather effects", "civil considerations"]},
    
    {"id": "F-03", "type": "factual", "ipb_step": 1,
     "question": "What is the difference between an area of operations and an area of interest as defined in ATP 2-01.3?",
     "keywords": ["area of operations", "assigned", "commander", "authority", "area of interest", "beyond", "influence", "objectives"]},
    
    {"id": "F-04", "type": "factual", "ipb_step": 3,
     "question": "What is the difference between a high-value target and a high-payoff target?",
     "keywords": ["high-value target", "enemy commander", "requires", "mission", "high-payoff target", "successful", "friendly", "course of action"]},
    
    {"id": "P-01", "type": "procedural", "ipb_step": 1,
     "question": "Walk through the substeps of IPB Step 1: Define the Operational Environment.",
     "keywords": ["limits", "area of operations", "area of interest", "significant characteristics", "evaluate", "holdings", "initiate", "collection"]},
    
    {"id": "P-02", "type": "procedural", "ipb_step": 2,
     "question": "How is a Modified Combined Obstacle Overlay (MCOO) constructed during IPB Step 2?",
     "keywords": ["MCOO", "obstacle", "terrain", "mobility corridors", "avenues of approach", "key terrain", "overlay", "combined"]},
    
    {"id": "P-03", "type": "procedural", "ipb_step": 4,
     "question": "What are the steps to develop threat courses of action during IPB Step 4?",
     "keywords": ["threat COA", "situation template", "event template", "event matrix", "HVT", "objectives", "likely"]},
    
    {"id": "P-04", "type": "procedural", "ipb_step": 0,
     "question": "How does the IPB process integrate with the Military Decision-Making Process?",
     "keywords": ["MDMP", "mission analysis", "COA development", "COA analysis", "war game", "intelligence estimate", "PIR"]},
    
    {"id": "C-01", "type": "comparative", "ipb_step": None,
     "question": "How does terrain analysis in IPB Step 2 differ from threat evaluation in IPB Step 3?",
     "keywords": ["terrain", "OAKOC", "effects", "operations", "threat", "capabilities", "doctrine", "tactics", "TTP"]},
    
    {"id": "C-02", "type": "comparative", "ipb_step": 3,
     "question": "How do regular, irregular, and hybrid threats differ according to ATP 2-01.3?",
     "keywords": ["regular", "military", "forces", "irregular", "unconventional", "guerrilla", "hybrid", "combination", "conventional", "asymmetric"]},
    
    {"id": "E-01", "type": "echelon_specific", "ipb_step": 0,
     "question": "How do IPB responsibilities differ between a BCT S-2 and a Division G-2?",
     "keywords": ["BCT", "S-2", "tactical", "division", "G-2", "operational", "scope", "detail", "resources"]},
    
    {"id": "E-02", "type": "echelon_specific", "ipb_step": 0,
     "question": "How do IPB products differ when produced at the tactical level versus the operational level?",
     "keywords": ["tactical", "operational", "detail", "scope", "area of interest", "depth", "products", "level"]},
    
    {"id": "AR-01", "type": "applied_reasoning", "ipb_step": 2,
     "question": "You are a BCT S-2 preparing IPB for an operation in a densely urbanized area against an irregular threat. How should your terrain analysis in Step 2 be modified compared to open terrain?",
     "keywords": ["urban", "three-dimensional", "OAKOC", "vertical", "subterranean", "buildings", "infrastructure", "civil", "MCOO", "population"]},
    
    {"id": "AR-02", "type": "applied_reasoning", "ipb_step": 3,
     "question": "You are evaluating a peer threat that possesses integrated air defense systems, electronic warfare capabilities, and long-range precision fires. How would you apply IPB Step 3 to characterize this threat?",
     "keywords": ["peer", "IADS", "electronic warfare", "precision fires", "A2/AD", "capabilities", "doctrine", "threat model", "template"]},
    
    {"id": "AR-03", "type": "applied_reasoning", "ipb_step": 4,
     "question": "Develop a situation template for a hybrid threat that combines conventional military forces with irregular militia and cyberspace operations in a contested urban environment.",
     "keywords": ["situation template", "hybrid", "conventional", "irregular", "militia", "cyberspace", "urban", "COA", "disposition"]},
    
    {"id": "AR-04", "type": "applied_reasoning", "ipb_step": 2,
     "question": "How would you conduct weather effects analysis during IPB Step 2 to support planning for an airborne operation?",
     "keywords": ["weather", "airborne", "wind", "visibility", "cloud ceiling", "precipitation", "drop zone", "effects matrix"]},
    
    {"id": "PG-01", "type": "product_generation", "ipb_step": 4,
     "question": "Create an initial high-value target list for a BCT operation against a mechanized infantry brigade in open terrain.",
     "keywords": ["HVT", "list", "commander", "requires", "mission", "command post", "artillery", "air defense", "logistics", "reconnaissance"]},
    
    {"id": "PG-02", "type": "product_generation", "ipb_step": 4,
     "question": "Describe the components and purpose of an event template and its associated event matrix.",
     "keywords": ["event template", "event matrix", "named area of interest", "NAI", "indicator", "activity", "time", "decision point"]},
    
    {"id": "CS-01", "type": "cross_step", "ipb_step": None,
     "question": "How do the outputs of IPB Step 1 (Define the OE) directly feed into and shape the analysis conducted in Step 2 (Describe Environmental Effects)?",
     "keywords": ["area of operations", "area of interest", "significant characteristics", "terrain analysis", "OAKOC", "effects", "scope"]},
    
    {"id": "CS-02", "type": "cross_step", "ipb_step": None,
     "question": "How does the threat evaluation from IPB Step 3 inform the development of threat courses of action in Step 4?",
     "keywords": ["threat model", "capabilities", "doctrine", "TTP", "COA", "situation template", "event template", "HVT"]},
    
    {"id": "MD-01", "type": "multi_domain", "ipb_step": 2,
     "question": "How should cyberspace domain effects be integrated into terrain analysis during IPB Step 2?",
     "keywords": ["cyberspace", "terrain", "network", "infrastructure", "effects", "friendly", "threat", "domain"]},
    
    {"id": "MD-02", "type": "multi_domain", "ipb_step": 3,
     "question": "What space domain threat capabilities should be considered during IPB Step 3 when evaluating a peer threat?",
     "keywords": ["space", "satellite", "reconnaissance", "GPS", "communications", "ASAT", "jamming", "peer"]},
    
    {"id": "CT-01", "type": "contrastive", "ipb_step": 4,
     "question": "Why does IPB Step 4 produce situation templates in addition to threat COA statements? What does the template provide that the statement alone cannot?",
     "keywords": ["situation template", "graphic", "spatial", "disposition", "COA statement", "text", "visualization", "wargame"]},
    
    {"id": "CT-02", "type": "contrastive", "ipb_step": 2,
     "question": "When should an analyst produce an MCOO versus relying on a terrain effects matrix alone? What does the MCOO provide that the matrix does not?",
     "keywords": ["MCOO", "overlay", "graphic", "spatial", "terrain effects matrix", "tabular", "combined", "visualization", "avenues"]}
]
```

### Scoring function
```python
def score_answer(answer: str, keywords: list[str]) -> float:
    """Score answer by keyword coverage."""
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords) if keywords else 0.0
```

---

## SECTION 10: PIPELINE ORCHESTRATOR INTERFACE

### CLI for run_pipeline.py
```
python run_pipeline.py                          # Run all stages sequentially
python run_pipeline.py --stages 1,2,3           # Run only generation stages
python run_pipeline.py --stages 4,5             # Run only training + eval
python run_pipeline.py --stages eval            # Eval only (requires existing adapter)
python run_pipeline.py --resume                  # Resume from last completed stage
python run_pipeline.py --target-count 5000       # Target QA count (default 5000)
python run_pipeline.py --adapter outputs/atp-gemma4-lora  # Specify adapter for eval
python run_pipeline.py --skip-enrichment         # Skip Stage 2 (use existing enriched.jsonl)
```

### Monitor interface (monitor.py)
```
Displays live progress:
- Stage currently executing
- Chunks processed / total
- QA pairs generated / target
- Quality filter pass rate
- Training loss (live from log)
- Estimated time remaining

Refresh: every 5 seconds
Data source: pipeline.log + data/*.jsonl file sizes
```

---

## SECTION 11: GGUF EXPORT

### burn_gguf.py interface (from FM 2-0 pipeline, adapted for Gemma 4)
```bash
python burn_gguf.py --adapter outputs/atp-gemma4-lora --quant q4_k_m

# Output: ./burns/atp-gemma4-lora/
#   ├── model.gguf
#   └── Modelfile

# Then register with Ollama:
cd burns/atp-gemma4-lora/
ollama create atp-ipb-gemma4 -f Modelfile
ollama run atp-ipb-gemma4
```

**Important:** Adapter was trained on Gemma 4 31B-it. The burn_gguf.py must read the base model from `adapter_config.json` dynamically — do NOT hardcode the base model path (FM 2-0 lesson: base model mismatch is a common inference error).

---

## SECTION 12: KNOWN GOTCHAS & ANTI-PATTERNS

1. **Never use fp16 with Gemma 4** — activations exceed fp16 max (65504). Always bf16.
2. **Never hardcode base model path** — read from adapter_config.json dynamically.
3. **Never use `"w"` mode for seeds file** — always append (`"a"`) for resumability.
4. **Never start answer with "According to ATP 2-01.3..."** — citations go at the END in a [Reference: ...] block.
5. **Never add beyond-manual content in metadata** — metadata is structural classification only.
6. **Never run Ollama and training simultaneously** — stop Ollama first, verify VRAM free.
7. **Never exceed 3 epochs at 5K+ examples** — overfitting risk. Start with 2 epochs.
8. **Never use LoRA r>16 without evidence it helps** — V2 proved r=32 overfits at this scale.
9. **Always validate ipb_step matches chapter number** — Chapter 3 = Step 1, Chapter 4 = Step 2, etc.
10. **Always use temperature 0.7 for generation, 0.1 for classification** — classification needs determinism.

---

## SECTION 13: EXECUTION CHECKLIST

```
[ ] 1. Pull Gemma 4 31B via Ollama: ollama pull gemma4:31b
[ ] 2. Verify PDF accessible: ls /mnt/project/ATP_2-01_3_*.pdf
[ ] 3. Create project directory: mkdir -p ~/caimll_finetuning/atp_pipeline_v2/data
[ ] 4. Extract PDF text: pdftotext <pdf> atp_raw.txt
[ ] 5. Run chunker → data/chunks.jsonl
[ ] 6. Run enricher → data/enriched.jsonl
[ ] 7. Validate metadata (spot-check 20 chunks manually)
[ ] 8. Run generator (target: 5000) → data/seeds.jsonl
[ ] 9. Run quality filter → count pass/fail rate
[ ] 10. Format for Gemma 4 → data/train.jsonl + data/val.jsonl
[ ] 11. Stop Ollama: ollama stop gemma4:31b
[ ] 12. Verify VRAM: nvidia-smi (should show ~0 used)
[ ] 13. Run trainer → outputs/atp-gemma4-lora/
[ ] 14. Check train loss (target: 1.0-1.2)
[ ] 15. Run eval against 24 questions → eval/results/
[ ] 16. If loss healthy + eval promising: scale to 10K, retrain
[ ] 17. GGUF export: python burn_gguf.py --adapter outputs/atp-gemma4-lora
[ ] 18. Register with Ollama: ollama create atp-ipb-gemma4 -f Modelfile
[ ] 19. Demo inference: ollama run atp-ipb-gemma4
```
