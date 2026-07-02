import re

def check_sql_safety(sql_content: str) -> tuple[bool, str | None]:
    """
    Safety check for dbt SQL models:
    A dbt model must only contain SELECT queries.
    This check scans the SQL text and rejects it if it contains dangerous commands
    like DROP, DELETE, TRUNCATE, ALTER, GRANT, INSERT, UPDATE, or CREATE OR REPLACE.
    """
    # Remove single-line comments (-- ...)
    sql_clean = re.sub(r'--.*$', '', sql_content, flags=re.MULTILINE)
    
    # Remove multi-line comments (/* ... */)
    sql_clean = re.sub(r'/\*[\s\S]*?\*/', '', sql_clean)
    
    # Tokenize and check for dangerous keywords
    tokens = re.findall(r'\b[a-zA-Z_]+\b', sql_clean.lower())
    
    dangerous_keywords = {
        "drop", "delete", "truncate", "alter", "grant", "insert", "update"
    }
    
    for token in tokens:
        if token in dangerous_keywords:
            return False, f"Dangerous SQL keyword detected: '{token.upper()}'."
            
    # Check for CREATE OR REPLACE statements
    if re.search(r'\bcreate\b\s+(?:or\s+replace\s+)?(?:table|view|procedure|function|model)\b', sql_clean.lower()):
        return False, "CREATE/REPLACE statement detected. A dbt model must contain SELECT queries only."
        
    return True, None
