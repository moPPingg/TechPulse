import sys
import json
from pathlib import Path
sys.path.append('d:/TechPulse')

try:
    from api import get_chart_data_v1
    res = get_chart_data_v1('FPT', 200)
    print("Function passed!")
    json.dumps(res)
    print("SUCCESS JSON DUMPS")
except Exception as e:
    import traceback
    traceback.print_exc()
