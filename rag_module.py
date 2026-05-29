"""
RAG Module — AI-Driven Log File Threat Detection System
--------------------------------------------------------
Uses ChromaDB + Sentence Transformers for retrieval
Uses Groq (LLaMA3) for human-readable threat explanation
"""

import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

# ── Constants ────────────────────────────────────────────────────────────────

# Attack type descriptions for context
ATTACK_DESCRIPTIONS = {
    'neptune':         'Neptune DoS attack — floods server with SYN packets to exhaust TCP connections',
    'smurf':           'Smurf DoS attack — amplified ICMP flood using broadcast addresses',
    'pod':             'Ping of Death — sends oversized ping packets to crash the target',
    'teardrop':        'Teardrop attack — sends fragmented IP packets to crash the OS',
    'land':            'LAND attack — sends packet with same source and destination IP to crash system',
    'back':            'Back attack — Apache vulnerability exploit causing DoS',
    'mscan':           'MScan port scanning — scans multiple ports to find vulnerabilities',
    'portsweep':       'Port sweep — scans a range of ports on a single target host',
    'nmap':            'Nmap reconnaissance — uses Nmap tool to scan and map network',
    'satan':           'SATAN probe — uses security analysis tool to find vulnerabilities',
    'ipsweep':         'IP sweep — scans multiple hosts to find active machines',
    'guess_passwd':    'Password guessing — brute force attack on login credentials',
    'ftp_write':       'FTP write attack — exploits FTP to write malicious files',
    'imap':            'IMAP vulnerability exploit — unauthorized mailbox access',
    'phf':             'PHF exploit — web CGI vulnerability allowing command execution',
    'multihop':        'Multi-hop attack — uses intermediate compromised hosts to attack',
    'warezmaster':     'Warezmaster — unauthorized FTP access to transfer pirated software',
    'warezclient':     'Warezclient — downloads pirated software via compromised FTP',
    'spy':             'Spy attack — unauthorized data collection from the system',
    'buffer_overflow': 'Buffer overflow — overwrites memory to gain root privileges',
    'loadmodule':      'Load module attack — loads unauthorized kernel modules for root access',
    'perl':            'Perl attack — exploits Perl setuid script for privilege escalation',
    'rootkit':         'Rootkit installation — installs hidden tools to maintain root access',
    'saint':           'SAINT probe — security audit tool used maliciously for reconnaissance',
    'normal':          'Normal traffic — legitimate network connection',
}

RISK_ACTIONS = {
    'High':   'IMMEDIATE ACTION REQUIRED — isolate affected system, block source, alert security team',
    'Medium': 'INVESTIGATE — monitor closely, check firewall rules, review access logs',
    'Low':    'MONITOR — log for pattern analysis, no immediate action needed',
}


# ── ChromaDB Setup ────────────────────────────────────────────────────────────

def build_knowledge_base():
    """
    Build ChromaDB vector store with NSL-KDD attack pattern knowledge.
    This is the RAG knowledge base — called once at startup.
    """
    print("[RAG] Building knowledge base...")

    # Initialize ChromaDB — stores in local folder 'chroma_db'
    client = chromadb.PersistentClient(path="./chroma_db")

    # Delete existing collection if it exists (fresh build)
    try:
        client.delete_collection("attack_patterns")
    except:
        pass

    collection = client.create_collection(
        name="attack_patterns",
        metadata={"hnsw:space": "cosine"}
    )

    # Load sentence transformer model for embeddings
    print("[RAG] Loading embedding model (first time may take 1-2 mins)...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # Knowledge base — attack patterns with context
    # Each document = one attack pattern with description and indicators
    documents = [
        # DoS Attacks
        "Neptune DoS attack detected. High serror_rate near 1.0. src_bytes is 0. flag is S0 meaning connection never established. count is very high above 200. This is a SYN flood attack targeting TCP services. Attacker sends thousands of SYN packets without completing handshake.",
        "Smurf DoS attack detected. Protocol is ICMP. High wrong_fragment count. Large number of packets from multiple sources. Amplification attack using broadcast address. Can generate massive traffic volume.",
        "Ping of Death attack detected. ICMP protocol. Oversized fragmented packets. wrong_fragment is high. Designed to crash or freeze target system.",

        # Probe Attacks
        "MScan port scanning detected. REJ flag indicating connection refused. diff_srv_rate is very high near 1.0 meaning many different services probed. duration is very short near 0. Attacker mapping open ports and services.",
        "Portsweep attack detected. Single destination host targeted. dst_host_count is high. dst_host_diff_srv_rate is high. Scanning all ports on one machine to find vulnerabilities.",
        "Nmap reconnaissance detected. REJ or RSTOS0 flags. Short duration connections. Multiple services targeted. Attacker building network map before launching main attack.",
        "IPSweep detected. Many different destination IPs. Short duration. ICMP or TCP protocol. Attacker scanning subnet to find active hosts.",

        # R2L Attacks
        "Password guessing attack detected. Service is ftp or ssh or telnet. num_failed_logins is high above 3. logged_in is 0 meaning still not authenticated. Brute force credential attack.",
        "FTP write attack detected. Service is ftp. logged_in is 1. num_file_creations is high. Attacker using FTP access to upload malicious files to server.",
        "IMAP exploit detected. Service is imap. Unauthorized mailbox access attempt. Remote to local attack pattern.",

        # U2R Attacks
        "Buffer overflow attack detected. root_shell is 1. su_attempted is 1. num_root is high. Attacker exploited memory vulnerability to gain root privileges. Critical severity.",
        "Rootkit installation detected. root_shell is 1. num_file_creations is high. num_shells is high. Attacker installing hidden tools to maintain persistent root access.",
        "Saint reconnaissance attack detected. Multiple services probed. diff_srv_rate is high. Attacker using SAINT security tool maliciously to map vulnerabilities.",

        # Normal
        "Normal network traffic. SF flag meaning connection completed normally. logged_in is 1. Reasonable src_bytes and dst_bytes. Low error rates. Standard service access.",
    ]

    # Generate embeddings
    print("[RAG] Generating embeddings...")
    embeddings = embedder.encode(documents).tolist()

    # Store in ChromaDB
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=[f"pattern_{i}" for i in range(len(documents))],
        metadatas=[{"type": "attack_pattern"} for _ in documents]
    )

    print(f"[RAG] Knowledge base built — {len(documents)} patterns stored")
    return collection, embedder


def load_knowledge_base():
    """Load existing ChromaDB knowledge base."""
    client = chromadb.PersistentClient(path="./chroma_db")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    collection = client.get_collection("attack_patterns")
    return collection, embedder


def get_knowledge_base():
    """Get or create knowledge base."""
    try:
        return load_knowledge_base()
    except:
        return build_knowledge_base()


# ── RAG Retrieval ─────────────────────────────────────────────────────────────

def retrieve_similar_patterns(query_text, collection, embedder, n_results=3):
    """
    Given a log description, retrieve similar attack patterns from ChromaDB.
    This is the R in RAG — Retrieval.
    """
    # Convert query to vector
    query_embedding = embedder.encode([query_text]).tolist()

    # Search ChromaDB for similar patterns
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )

    return results['documents'][0]  # Return top matching documents


# ── Prompt Engineering ────────────────────────────────────────────────────────

def build_prompt(log_features, prediction, probability, risk_level,
                 attack_label, similar_patterns):
    """
    Build a structured prompt for the LLM.
    This is the prompt engineering step.
    """

    # Get attack description
    attack_desc = ATTACK_DESCRIPTIONS.get(
        str(attack_label).lower(),
        f"Unknown attack type: {attack_label}"
    )

    # Format similar patterns as context
    context = "\n".join([f"- {p}" for p in similar_patterns])

    # Format key features
    key_features = f"""
- Protocol: {log_features.get('protocol_type', 'unknown')}
- Service: {log_features.get('service', 'unknown')}
- Flag: {log_features.get('flag', 'unknown')}
- Source Bytes: {log_features.get('src_bytes', 0)}
- Destination Bytes: {log_features.get('dst_bytes', 0)}
- Failed Logins: {log_features.get('num_failed_logins', 0)}
- Logged In: {log_features.get('logged_in', 0)}
- Error Rate: {log_features.get('serror_rate', 0)}
- Same Service Rate: {log_features.get('same_srv_rate', 0)}
- Connection Count: {log_features.get('count', 0)}"""

    prompt = f"""You are a cybersecurity analyst. Analyze this network log entry and provide a clear, professional threat assessment.

DETECTED THREAT:
- Classification: {prediction}
- Attack Type: {attack_label}
- Risk Level: {risk_level}
- Confidence: {probability:.1%}
- Attack Description: {attack_desc}

KEY NETWORK FEATURES:
{key_features}

SIMILAR HISTORICAL PATTERNS FROM KNOWLEDGE BASE:
{context}

Provide a threat explanation in EXACTLY this format:

THREAT SUMMARY: (1 sentence — what type of attack this is)
ATTACK BEHAVIOR: (1-2 sentences — what the attacker is doing based on the features)
RISK ASSESSMENT: (1 sentence — why this risk level was assigned)
RECOMMENDED ACTION: (1-2 sentences — what the security team should do immediately)

Keep the response concise, technical, and actionable. Do not include any extra text outside these 4 sections."""

    return prompt


# ── LLM Explanation ───────────────────────────────────────────────────────────

def generate_explanation(prompt, api_key):
    """
    Send prompt to Groq LLM and get threat explanation.
    This is the G in RAG — Generation.
    """
    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",   # Current Groq model (llama3-8b-8192 decommissioned)
        messages=[
            {
                "role": "system",
                "content": "You are a professional cybersecurity analyst. Provide concise, accurate, actionable threat assessments."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=300,
        temperature=0.3   # Low temperature = more consistent, factual output
    )

    return response.choices[0].message.content


# ── Hallucination Mitigation ──────────────────────────────────────────────────

def validate_explanation(explanation, prediction, attack_label):
    """
    Cross-verify LLM output against model prediction.
    Hallucination mitigation — ensures LLM doesn't contradict the ML model.
    """
    explanation_lower = explanation.lower()
    prediction_lower = prediction.lower()

    # Check 1 — if model says Normal, LLM should not say attack
    if prediction_lower == 'normal':
        attack_words = ['attack', 'malicious', 'threat', 'exploit', 'flood']
        for word in attack_words:
            if word in explanation_lower:
                return False, "LLM contradicts model — flagged as attack but model says Normal"

    # Check 2 — if model says Malicious, LLM should not say normal/safe
    if prediction_lower == 'malicious':
        safe_words = ['normal traffic', 'legitimate', 'safe', 'no threat']
        for phrase in safe_words:
            if phrase in explanation_lower:
                return False, "LLM contradicts model — flagged as safe but model says Malicious"

    # Check 3 — explanation must have the required sections
    required_sections = ['THREAT SUMMARY', 'ATTACK BEHAVIOR',
                        'RISK ASSESSMENT', 'RECOMMENDED ACTION']
    for section in required_sections:
        if section not in explanation:
            return False, f"LLM response missing section: {section}"

    return True, "Validated"


# ── Main RAG Function ─────────────────────────────────────────────────────────

def get_threat_explanation(log_row, prediction, probability,
                           risk_level, attack_label, api_key,
                           collection, embedder):
    """
    Complete RAG pipeline for one log entry.
    Returns human-readable threat explanation.
    """

    # Step 1 — Build query from log features
    query = f"{attack_label} attack {prediction} {risk_level} risk protocol {log_row.get('protocol_type', '')} service {log_row.get('service', '')} flag {log_row.get('flag', '')}"

    # Step 2 — Retrieve similar patterns (RAG retrieval)
    similar_patterns = retrieve_similar_patterns(query, collection, embedder)

    # Step 3 — Build prompt (prompt engineering)
    prompt = build_prompt(
        log_features=log_row,
        prediction=prediction,
        probability=probability,
        risk_level=risk_level,
        attack_label=attack_label,
        similar_patterns=similar_patterns
    )

    # Step 4 — Generate explanation (LLM)
    try:
        explanation = generate_explanation(prompt, api_key)
    except Exception as e:
        return f"LLM Error: {str(e)}", False

    # Step 5 — Validate (hallucination mitigation)
    is_valid, validation_msg = validate_explanation(
        explanation, prediction, attack_label
    )

    if not is_valid:
        # Return fallback explanation if LLM hallucinated
        fallback = f"""THREAT SUMMARY: {ATTACK_DESCRIPTIONS.get(str(attack_label).lower(), 'Unknown attack')}
ATTACK BEHAVIOR: Model detected {prediction} traffic with {probability:.1%} confidence based on network feature analysis.
RISK ASSESSMENT: Classified as {risk_level} risk based on attack probability score.
RECOMMENDED ACTION: {RISK_ACTIONS.get(risk_level, 'Review and investigate')}"""
        return fallback, False

    return explanation, True


# ── Batch Processing ──────────────────────────────────────────────────────────

def get_explanations_for_top_threats(result_df, original_df,
                                      api_key, collection, embedder,
                                      max_explanations=5):
    """
    Get RAG explanations for top N high-risk entries only.
    We limit to 5 to avoid Groq rate limits.
    """
    explanations = {}

    # Filter to high risk malicious entries only
    high_risk = result_df[
        (result_df['Prediction'] == 'Malicious') &
        (result_df['Risk Level'] == 'High')
    ].head(max_explanations)

    if len(high_risk) == 0:
        # Fall back to medium risk if no high risk
        high_risk = result_df[
            result_df['Prediction'] == 'Malicious'
        ].head(max_explanations)

    for idx in high_risk.index:
        try:
            # Get original log row features
            if idx < len(original_df):
                log_row = original_df.iloc[idx].to_dict()
            else:
                log_row = {}

            prediction   = result_df.loc[idx, 'Prediction']
            probability  = float(result_df.loc[idx, 'Attack Probability'].replace('%','')) / 100
            risk_level   = result_df.loc[idx, 'Risk Level']
            attack_label = result_df.loc[idx].get('Original Label', 'unknown')

            explanation, is_valid = get_threat_explanation(
                log_row=log_row,
                prediction=prediction,
                probability=probability,
                risk_level=risk_level,
                attack_label=attack_label,
                api_key=api_key,
                collection=collection,
                embedder=embedder
            )

            explanations[idx] = {
                'explanation': explanation,
                'validated': is_valid,
                'log_num': result_df.loc[idx, 'Log #']
            }

        except Exception as e:
            explanations[idx] = {
                'explanation': f"Could not generate explanation: {str(e)}",
                'validated': False,
                'log_num': idx + 1
            }

    return explanations


if __name__ == "__main__":
    # Test the RAG module standalone
    print("Testing RAG module...")
    collection, embedder = build_knowledge_base()
    print("Knowledge base ready!")

    # Test retrieval
    patterns = retrieve_similar_patterns(
        "neptune attack high risk serror_rate 1.0",
        collection, embedder
    )
    print(f"\nRetrieved {len(patterns)} similar patterns:")
    for p in patterns:
        print(f"  - {p[:80]}...")
