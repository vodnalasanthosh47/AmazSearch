import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

# ⚠️  load_dotenv MUST run BEFORE importing agents — first_agent.py reads
#     LLM_TO_USE from os.environ at import time.
_env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=_env_path)

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# pyrefly: ignore [missing-import]
from first_agent import first_agent, infer_products_from_vague_input, InferredSetupSpecs
# pyrefly: ignore [missing-import]
from spec_resolution_agent import spec_resolution_agent, resolve_specs
# pyrefly: ignore [missing-import]
from product_search_agent import query_builder_agent, search_products
# pyrefly: ignore [missing-import]
from compliance_check_agent import compliance_check_agent, check_compliance

APP_NAME = "amazesearch_app"
USER_ID = "demo_user"
SESSION_ID = "demo_session"
SPEC_SESSION_ID = "demo_session_spec_resolution"
SEARCH_SESSION_ID = "demo_session_product_search"
COMPLIANCE_SESSION_ID = "demo_session_compliance_check"


async def main():
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SPEC_SESSION_ID
    )
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SEARCH_SESSION_ID
    )
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=COMPLIANCE_SESSION_ID
    )

    # Runner for first_agent (the refinement loop)
    first_agent_runner = Runner(
        app_name=APP_NAME,
        agent=first_agent,
        session_service=session_service,
    )

    # Runner for spec_resolution_agent (search + structure pipeline)
    spec_runner = Runner(
        app_name=APP_NAME,
        agent=spec_resolution_agent,
        session_service=session_service,
    )

    # Runner for query_builder_agent (builds search queries; amazon_search is called programmatically)
    search_runner = Runner(
        app_name=APP_NAME,
        agent=query_builder_agent,
        session_service=session_service,
    )

    # Runner for compliance_check_agent (detail + compliance pipeline)
    compliance_runner = Runner(
        app_name=APP_NAME,
        agent=compliance_check_agent,
        session_service=session_service,
    )

    print("Describe what you're setting up (e.g. 'beginner gym equipment').")
    print("Keep refining as needed. Type 'done' once you're satisfied with the list.\n")

    last_specs: InferredSetupSpecs | None = None

    while True:
        user_prompt = input("You: ").strip()

        if not user_prompt:
            continue

        if user_prompt.lower() in {"done", "exit", "quit", "q"}:
            break

        try:
            last_specs = await infer_products_from_vague_input(
                first_agent_runner, USER_ID, SESSION_ID, user_prompt
            )
        except Exception as e:
            print(f"\n[Error generating specs: {e}]\n")
            continue

        print("\nGuessed Setup Structure:")
        print(last_specs.model_dump_json(indent=2))
        print()

    if last_specs is None:
        print("\nNo confirmed product list to resolve. Exiting.")
        return

    # ── Stage 2: Resolve real-world specs ──────────────────────────────────
    print("\nResolving real-world specs and brands for your confirmed list...\n")

    try:
        resolved = await resolve_specs(
            spec_runner, USER_ID, SPEC_SESSION_ID, last_specs
        )
    except Exception as e:
        print(f"\n[Error resolving specs: {e}]")
        return

    print("Resolved Specs:")
    print(resolved.model_dump_json(indent=2))
    print()

    # ── Stage 3: Search Amazon for matching products ───────────────────────
    print("Searching Amazon for matching products...\n")

    try:
        search_results = await search_products(
            search_runner, USER_ID, SEARCH_SESSION_ID, resolved
        )
    except Exception as e:
        print(f"\n[Error searching products: {e}]")
        return

    total_products = sum(len(g.products) for g in search_results.items)
    print(f"Found {total_products} products across {len(search_results.items)} categories.")
    print(search_results.model_dump_json(indent=2))
    print()

    # ── Stage 4: Check compliance against specs ────────────────────────────
    print("Checking product compliance against specs...\n")

    try:
        compliant = await check_compliance(
            compliance_runner, USER_ID, COMPLIANCE_SESSION_ID, search_results
        )
    except Exception as e:
        print(f"\n[Error checking compliance: {e}]")
        return

    total_compliant = sum(len(g.compliant_products) for g in compliant.items)
    print(f"\n✅ {total_compliant} compliant products found!\n")
    print("Final Compliant Products:")
    print(compliant.model_dump_json(indent=2))


if __name__ == "__main__":
    # Ensure your environment variable is set in backend/.env: GOOGLE_API_KEY="your-key"
    asyncio.run(main())
