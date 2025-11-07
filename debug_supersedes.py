import json
import re
from datetime import datetime
from src.db import get_db_connection
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def normalize_str(s: str) -> str:
    if not s:
        return ""
    s = s.replace("'", "'").replace("'", "'")
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_paths(paths):
    if not paths:
        return []
    if isinstance(paths, list) and paths and isinstance(paths[0], list):
        return paths
    if isinstance(paths, list) and paths and isinstance(paths[0], str):
        return [paths]
    if isinstance(paths, str):
        try:
            parsed = json.loads(paths)
            if isinstance(parsed, list):
                if parsed and isinstance(parsed[0], list):
                    return parsed
                if parsed and isinstance(parsed[0], str):
                    return [parsed]
        except Exception:
            pass
    return []

def category_matches(target_paths, candidate_paths):
    for t in target_paths:
        for c in candidate_paths:
            if not isinstance(t, list) or not isinstance(c, list):
                continue
            t_norm = [p.lower() for p in t]
            c_norm = [p.lower() for p in c]
            if len(c_norm) <= len(t_norm) and t_norm[:len(c_norm)] == c_norm:
                return True
            if len(t_norm) <= len(c_norm) and c_norm[:len(t_norm)] == t_norm:
                return True
    return False

def debug_implicit_supersedes():
    # Target document - CHRO's Circular No. 03-2024
    target_file = "CHRO's Circular No. 03-2024"
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Get target document
            cur.execute("SELECT filename, metadata FROM document_metadata WHERE filename = %s", (target_file,))
            target_row = cur.fetchone()
            
            if not target_row:
                print(f"Target document {target_file} not found!")
                return
                
            target_metadata = target_row[1] if isinstance(target_row[1], dict) else json.loads(target_row[1])
            
            print(f"=== TARGET DOCUMENT: {target_file} ===")
            print(f"Circular Number: {target_metadata.get('circular_number')}")
            print(f"Issued Date: {target_metadata.get('issued_date')}")
            print(f"Department: {target_metadata.get('department')}")
            print(f"Category Path: {target_metadata.get('category_path')}")
            print()
            
            # Parse target document details
            cur_issued = target_metadata.get("issued_date")
            cur_dt = None
            for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
                try:
                    cur_dt = datetime.strptime(cur_issued, fmt)
                    break
                except Exception:
                    pass
            
            raw_category_paths = target_metadata.get("category_path") or []
            category_paths = normalize_paths(raw_category_paths)
            cur_dept = normalize_str(target_metadata.get("department"))
            
            print(f"Parsed target details:")
            print(f"  Date: {cur_dt}")
            print(f"  Department (normalized): '{cur_dept}'")
            print(f"  Category paths (normalized): {category_paths}")
            print()
            
            # Get all other documents
            cur.execute("SELECT filename, metadata FROM document_metadata")
            rows = cur.fetchall()
            
            candidates = []
            rejected = []
            
            print("=== EVALUATING ALL CANDIDATES ===")
            for filename, row_meta in rows:
                if normalize_str(filename) == normalize_str(target_file):
                    print(f"SKIP: {filename} (self)")
                    continue
                    
                try:
                    other = row_meta if isinstance(row_meta, dict) else json.loads(row_meta)
                except Exception:
                    print(f"SKIP: {filename} (malformed metadata)")
                    continue
                
                other_cn = other.get("circular_number") or other.get("source_file")
                other_issued = other.get("issued_date")
                other_paths_raw = other.get("category_path") or []
                other_paths = normalize_paths(other_paths_raw)
                other_dept = normalize_str(other.get("department"))
                
                print(f"\n--- CANDIDATE: {filename} ---")
                print(f"  Circular Number: {other_cn}")
                print(f"  Issued Date: {other_issued}")
                print(f"  Department: {other.get('department')} -> normalized: '{other_dept}'")
                print(f"  Category Path Raw: {other_paths_raw}")
                print(f"  Category Path Normalized: {other_paths}")
                
                reasons = []
                
                # Date presence
                if not other_issued:
                    reasons.append("missing issued_date")
                
                # Category presence
                if not other_paths:
                    reasons.append("missing category_path")
                
                # Date parse
                other_dt = None
                if other_issued:
                    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
                        try:
                            other_dt = datetime.strptime(other_issued, fmt)
                            break
                        except Exception:
                            pass
                    if other_dt is None:
                        reasons.append(f"bad issued_date_format ({other_issued})")
                
                # Older check
                if other_dt and other_dt >= cur_dt:
                    reasons.append(f"not older (other_dt={other_dt.date()} >= cur_dt={cur_dt.date()})")
                
                # Category match
                cat_match = False
                if other_paths:
                    cat_match = category_matches(category_paths, other_paths)
                    if not cat_match:
                        reasons.append("category_mismatch")
                        print(f"  Category match details:")
                        print(f"    Target paths: {category_paths}")
                        print(f"    Candidate paths: {other_paths}")
                
                # Department checks
                if not cur_dept or not other_dept:
                    reasons.append("missing department")
                elif cur_dept != other_dept:
                    reasons.append(f"dept_mismatch ({other_dept} != {cur_dept})")
                
                print(f"  Parsed Date: {other_dt}")
                print(f"  Category Match: {cat_match}")
                print(f"  Reasons for rejection: {reasons}")
                
                if reasons:
                    rejected.append((filename, other_cn, reasons))
                    print(f"  RESULT: REJECTED")
                else:
                    candidates.append(other_cn)
                    print(f"  RESULT: ACCEPTED")
            
            print(f"\n=== SUMMARY ===")
            print(f"Accepted candidates: {candidates}")
            print(f"\nRejected candidates:")
            for filename, cn, reasons in rejected:
                print(f"  {cn} ({filename}): {', '.join(reasons)}")
                
    finally:
        conn.close()

if __name__ == "__main__":
    debug_implicit_supersedes()
