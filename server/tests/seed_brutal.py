"""Seed the real roster with a large, deliberately nasty set of agents for retrieval stress tests.

    python -m server.tests.seed_brutal          # add ~230 agents (idempotent)
    python -m server.tests.seed_brutal --clear  # remove only what this script added

Every summary ends in "[seed2]" so the rows are easy to find and remove. Run from the repo root.
"""

from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta, timezone

from server.services.execution.people import PersonRef
from server.services.execution.roster import get_agent_roster

TAG = "[seed2]"
random.seed(7)
roster = get_agent_roster()
backdate: dict[str, float] = {}  # agent -> days ago


def clear() -> None:
    with roster._tx() as conn:  # noqa: SLF001 - test tooling
        names = [r[0] for r in conn.execute("SELECT name FROM agents WHERE summary LIKE ?", (f"%{TAG}",))]
        for table in ("agent_people", "agent_threads", "agent_drafts"):
            conn.executemany(f"DELETE FROM {table} WHERE agent_name = ?", [(n,) for n in names])
        conn.executemany("DELETE FROM agents WHERE name = ?", [(n,) for n in names])
    print(f"removed {len(names)} agents")


if "--clear" in sys.argv:
    clear()
    sys.exit(0)


def agent(name, summary, subjects=(), people=(), days_ago=None, drafts=(), threads=()):
    roster.add_agent(name)
    roster.record_activity(
        name,
        [PersonRef(email=e.lower(), name=n, role="recipient") for e, n in people],
        list(threads),
        list(subjects),
    )
    roster.set_summary(name, f"{summary} {TAG}")
    for i, (recipient, subject) in enumerate(drafts):
        roster.open_draft(name, f"{name[:6]}-d{i}", recipient, subject)
    if days_ago is not None:
        backdate[name] = days_ago


def person_agent(first, last, email, topic, days_ago=None, extra_people=()):
    agent(
        f"Email to {first} {last}",
        f"Asked {first} about {topic.lower()}; waiting for a reply.",
        [topic],
        [(email, f"{first} {last}"), *extra_people],
        days_ago,
    )


# 1. Same first name, many last names, different topics ------------------------------------------
for first, lasts in {
    "Sarah": ["Lee", "Cohen", "Okafor", "Mitchell", "Tanaka", "Dubois", "Novak", "Reyes"],
    "Michael": ["Chen", "Rossi", "Adeyemi", "Walsh", "Petrov", "Haddad"],
    "David": ["Kim", "Park", "Singh", "Larsen", "Moreau", "Ortiz", "Brandt"],
    "Emma": ["Watson", "Clark", "Ivanova", "Santos", "Fischer"],
}.items():
    topics = ["Quarterly Review", "Wedding RSVP", "Apartment Sublet", "Project Handoff", "Conference Talk",
              "Invoice 4471", "Tax Documents", "Birthday Dinner"]
    for i, last in enumerate(lasts):
        person_agent(first, last, f"{first.lower()}.{last.lower()}@{random.choice(['gmail.com','outlook.com','acme.io','proton.me'])}",
                     topics[i % len(topics)], days_ago=random.choice([0.1, 0.5, 2, 6, 20, 45]))

# 2. Same last name, many first names ---------------------------------------------------------------
for last in ("Kim", "Patel", "Nguyen", "Garcia"):
    for first in ("Aaron", "Bianca", "Callum", "Diya", "Ezra", "Farah"):
        person_agent(first, last, f"{first.lower()}.{last.lower()}@work{random.randint(1,9)}.com",
                     random.choice(["Onboarding", "Offsite Planning", "Expense Report", "Reference Letter"]),
                     days_ago=random.choice([0.2, 1, 4, 12, 30]))

# 3. TRUE duplicates: identical full name, different people/emails ----------------------------------
agent("Email to Michael Chen (Stripe)", "Recruiter Michael Chen at Stripe about the backend offer.",
      ["Backend Engineer offer - Stripe"], [("michael.chen@stripe.com", "Michael Chen")], 1)
agent("Email to Michael Chen (landlord)", "Landlord Michael Chen about the broken dishwasher.",
      ["Dishwasher repair"], [("mchen.properties@gmail.com", "Michael Chen")], 3)
agent("Email to Michael Chen (college)", "Old roommate Michael Chen, planning the reunion.",
      ["Reunion 2026"], [("mike.chen92@yahoo.com", "Michael Chen")], 40)
agent("Email to Priya Sharma (Figma)", "Hiring manager Priya Sharma at Figma.", ["Design Eng interview"],
      [("priya@figma.com", "Priya Sharma")], 2)
agent("Email to Priya Sharma (dentist)", "Dentist office manager Priya Sharma, rescheduling cleaning.",
      ["Cleaning reschedule"], [("priya.sharma@smilecare.ca", "Priya Sharma")], 8)

# 4. Nickname / alias collisions ----------------------------------------------------------------------
agent("Email to Robert Johnson", "Robert Johnson, accountant, about 2025 taxes.", ["2025 tax filing"],
      [("bob.johnson@cpa-group.com", "Robert Johnson")], 5)
agent("Email to Bob Marley Jr", "Bob from the climbing gym about a membership freeze.", ["Membership freeze"],
      [("bob@vertical-gym.com", "Bob Marley Jr")], 2)
agent("Email to Katherine Lowe", "Kate Lowe, manager, 1:1 scheduling.", ["1:1 Thursday"],
      [("kate.lowe@corp.com", "Katherine Lowe")], 1)
agent("Email to Kathy Lowe-Brandt", "Kathy, aunt, Thanksgiving hosting.", ["Thanksgiving"],
      [("kathy.lb@aol.com", "Kathy Lowe-Brandt")], 10)
agent("Email to José García", "José García about the soccer league sign-up.", ["Soccer league"],
      [("jose.garcia@liga.mx", "José García")], 3)
agent("Email to Zoë O'Brien", "Zoë O'Brien about the book club pick.", ["Book club: November"],
      [("zoe.obrien@gmail.com", "Zoë O'Brien")], 4)
agent("Email to Jean-Luc Picard-Smith", "Jean-Luc about the wine tasting event.", ["Wine tasting"],
      [("jl.picardsmith@example.fr", "Jean-Luc Picard-Smith")], 15)

# 5. Shared mailbox on many agents: must NOT make them tie --------------------------------------------
for i in range(15):
    agent(f"Vendor Quote {i + 1}", f"Collected quote #{i + 1} from vendor {i + 1}; sales desk cc'd.",
          [f"Quote {i + 1}"], [("sales@bigvendor.com", "Sales Desk"), (f"rep{i + 1}@vendor{i + 1}.com", f"Rep {i + 1}")],
          days_ago=random.choice([1, 5, 9, 14]))

# 6. Near-identical topics (the hard similar-sounding ones) ---------------------------------------------
similar = {
    "Apartment lease": ["Maple St", "Queen West", "Harbourfront", "King Street", "Dundas Loft"],
    "Flight booking": ["Tokyo", "Toronto", "Tampa", "Tel Aviv", "Tallinn"],
    "Job offer": ["Vercel", "Stripe", "Figma", "Notion", "Linear", "Ramp"],
    "Q budget": ["Q1", "Q2", "Q3", "Q4"],
    "Doctor appointment": ["dentist", "dermatologist", "optometrist", "physio", "vet for Biscuit"],
    "Car": ["insurance renewal", "oil change", "parking ticket", "lease transfer", "winter tires"],
    "Python": ["course signup", "conference talk", "meetup RSVP", "package release", "bug report"],
}
for head, tails in similar.items():
    for t in tails:
        agent(f"{head} - {t}", f"{head} thread regarding {t}; awaiting confirmation.", [f"{head} {t}"],
              [(f"{head.split()[0].lower()}-{t.split()[0].lower()}@example.com", f"{t} contact")],
              days_ago=random.choice([0.3, 2, 7, 21, 60]))

# 7. Wildly random topics --------------------------------------------------------------------------------
random_topics = [
    ("Sourdough Starter Club", "Swapping starter with neighbours.", "Starter swap Sunday"),
    ("Marathon Training Plan", "Coach sent a 16-week plan; needs shoes.", "16-week plan"),
    ("Passport Renewal", "Passport office appointment, form DS-82.", "Passport renewal"),
    ("Kitchen Renovation", "Contractor Dave Ruiz quoting cabinets.", "Cabinet quote"),
    ("Grandma's 90th", "Organising surprise party, caterer booked.", "Surprise party"),
    ("Cat Insurance", "Comparing pet insurance for Biscuit.", "Pet insurance quote"),
    ("Hackathon Team", "Recruiting teammates for the weekend hackathon.", "Hackathon signup"),
    ("Photo Backup", "Moving 200GB of photos to cold storage.", "Backup plan"),
    ("Tax Return 2025", "Gathering T4s and receipts.", "T4 slips"),
    ("Charity 5K", "Registered for the Heart & Stroke 5K.", "5K registration"),
    ("Conference Travel - NeurIPS", "Booking hotel and flights for NeurIPS.", "NeurIPS travel"),
    ("Mortgage Pre-approval", "Broker needs pay stubs.", "Pre-approval documents"),
    ("Piano Lessons", "Booking a teacher, Tuesdays.", "Lesson schedule"),
    ("Moving Truck", "Quotes from three movers.", "Moving quote"),
    ("Wedding Gift - Alex & Sam", "Registry shopping.", "Registry"),
    ("Gym Cancellation", "Cancel Fit4Less before the next billing date.", "Cancellation request"),
    ("Newsletter Digest", "Summaries of Substack newsletters.", "Weekly digest"),
    ("Lost Luggage Claim", "Air Canada claim reference AC-88231.", "Baggage claim"),
    ("Jury Duty Deferral", "Deferral form submitted.", "Jury summons"),
    ("Concert Tickets", "Resale tickets for the June show.", "Ticket transfer"),
]
for name, summary, subject in random_topics:
    agent(name, summary, [subject], days_ago=random.choice([0.5, 3, 9, 25, 70]))

# 8. Open drafts / open loops on a handful (follow-up magnets) ---------------------------------------------
agent("Draft - Resignation Letter", "Drafted resignation letter, not sent.", ["Notice of resignation"],
      [("hr@corp.com", "HR")], 0.2, drafts=[("hr@corp.com", "Notice of resignation")])
agent("Draft - Apology to Landlord", "Drafted an apology about the late rent.", ["Late rent apology"],
      [("tom.hale@rentco.com", "Tom Hale")], 1, drafts=[("tom.hale@rentco.com", "Late rent apology")])
agent("Draft - Birthday Wishes", "Drafted birthday note to Sarah Lee.", ["Happy birthday"],
      [("sarah.lee@gmail.com", "Sarah Lee")], 3, drafts=[("sarah.lee@gmail.com", "Happy birthday")])

# 9. Misleading names (agent name disagrees with the people it actually handles) -------------------------------
agent("Email to Tom", "Actually corresponding with Alice Whitfield about the grant application.",
      ["Grant application"], [("alice.whitfield@foundation.org", "Alice Whitfield")], 1)
agent("Misc", "Everything about Greg Palmer's consulting invoice.", ["Invoice 2209"],
      [("greg@palmerconsulting.com", "Greg Palmer")], 2)
agent("Untitled", "Thread with the Zillow listing agent about 12 Elm St.", ["12 Elm St showing"],
      [("showings@elmrealty.com", "Elm Realty")], 6)

# 10. Stale agents (should rank low unless named) -----------------------------------------------------------------
for i in range(20):
    agent(f"Archive Task {i}", f"Old one-off lookup {i}.", [f"Old lookup {i}"], days_ago=random.choice([90, 120, 200]))

# 11. Plus-addressing / lookalike emails ----------------------------------------------------------------------------
agent("Email to Alex Rivera (work)", "Alex Rivera at work.", ["Sprint planning"],
      [("alex.rivera@corp.com", "Alex Rivera")], 1)
agent("Email to Alex Rivera (personal)", "Alex Rivera personal address, climbing trip.", ["Climbing trip"],
      [("alex.rivera+climb@gmail.com", "Alex Rivera")], 2)
agent("Email to Alex Rivero", "Alex Rivero, a different person, about the sublet.", ["Sublet"],
      [("alex.rivero@outlook.com", "Alex Rivero")], 2)


def apply_backdating() -> None:
    now = datetime.now(timezone.utc)
    with roster._tx() as conn:  # noqa: SLF001 - test tooling
        for name, days in backdate.items():
            stamp = (now - timedelta(days=days)).isoformat(timespec="seconds")
            conn.execute("UPDATE agents SET last_active_at = ? WHERE name = ?", (stamp, name))


apply_backdating()
print(f"roster now has {roster.count()} agents")
