from app.services.planner import DepartmentProposal, GeneratedPlan, MayorBaseline, _enforce_spending_cap


def test_spending_cap_enforced_with_scaling():
    baseline = MayorBaseline(mayor_total_budget=100.0, deficit_context="deficit")
    plan = GeneratedPlan(
        executive_summary="x",
        strategic_priorities="x",
        department_strategy="x",
        deficit_handling="x",
        risks_tradeoffs="x",
        generated_total_budget=120.0,
        departments=[
            DepartmentProposal(
                department="Health",
                proposed_fy_2025_26=50.0,
                proposed_fy_2026_27=70.0,
                proposed_directional="increase",
                rationale="test",
            )
        ],
    )

    adjusted, ok = _enforce_spending_cap(plan, baseline)
    assert ok
    assert adjusted.generated_total_budget == 100.0
    assert adjusted.departments[0].proposed_fy_2026_27 < 70.0
