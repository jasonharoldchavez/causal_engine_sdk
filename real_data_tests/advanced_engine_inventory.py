import sys
sys.path.insert(0,'.')

print("=== ADVANCED ENGINE IMPORT TEST ===")

tests = []

try:
    from advanced_math.geometry.violet_engine import VioletEngine
    tests.append("VioletEngine OK")
except Exception as e:
    tests.append("VioletEngine FAIL "+str(e))

try:
    from advanced_math.geometry.takens_embedding import TopologicalStateEngine
    tests.append("Takens/TopologicalStateEngine OK")
except Exception as e:
    tests.append("Takens FAIL "+str(e))

try:
    from advanced_math.topology.hodge_engine import ASIEHodgeEngine
    tests.append("HodgeEngine OK")
except Exception as e:
    tests.append("Hodge FAIL "+str(e))

try:
    from advanced_math.topology.persistent_homology import ASIETopologicalEngine
    tests.append("PersistentHomology OK")
except Exception as e:
    tests.append("PersistentHomology FAIL "+str(e))

try:
    from advanced_math.analytics.advanced_asie_engine import AdvancedASIEEngine
    tests.append("AdvancedASIE OK")
except Exception as e:
    tests.append("AdvancedASIE FAIL "+str(e))

try:
    from causal_engine.models.var_models import PluginLinearVAR
    tests.append("Causal VAR OK")
except Exception as e:
    tests.append("Causal VAR FAIL "+str(e))

try:
    from advanced_math.analytics.state_vector_asie import StateVectorASIE
    tests.append("StateVectorASIE OK")
except Exception as e:
    tests.append("StateVector FAIL "+str(e))

for t in tests:
    print(t)

print("=== COMPLETE ===")
