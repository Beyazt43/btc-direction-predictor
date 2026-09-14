"""Read-only API and dashboard.

Every write path -- training, activation, drift checks -- lives on the CLI and
the scheduler. This service can observe the system but not change it, which
keeps context.md §3's separation honest: a bug or a bad actor on the web
surface cannot retrain or roll back anything.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from btcpred.api import queries
from btcpred.config import get_settings
from btcpred.db.session import get_engine, get_sessionmaker
from btcpred.ingest.binance import interval_to_timedelta
from btcpred.predict.service import MODEL_NAMES

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    await get_engine().dispose()


app = FastAPI(
    title="BTC/USD next-hour direction predictor",
    description=(
        "Read-only view of a live ARIMA-vs-XGBoost comparison. Every number is "
        "computed from live, resolved predictions only; nothing here is back-filled."
    ),
    lifespan=lifespan,
)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with get_sessionmaker()() as session:
        yield session


def get_interval() -> timedelta:
    return interval_to_timedelta(get_settings().binance_interval)


Session = Annotated[AsyncSession, Depends(get_session)]
Interval = Annotated[timedelta, Depends(get_interval)]
Days = Annotated[int | None, Query(ge=1, le=3650, description="window in days; omit for all")]


def _check_model(model: str) -> str:
    if model not in MODEL_NAMES:
        raise HTTPException(404, f"unknown model {model!r}; expected one of {MODEL_NAMES}")
    return model


@app.get("/health")
async def health(session: Session) -> dict[str, Any]:
    return await queries.health(session, get_settings().binance_symbol)


@app.get("/metrics/live")
async def metrics_live(session: Session, interval: Interval) -> dict[str, Any]:
    """Live accuracy per model over §8's windows, with baselines and the CI."""
    return {
        name: [w.to_dict() for w in await queries.live_metrics(session, name, interval)]
        for name in MODEL_NAMES
    }


@app.get("/metrics/live/{model}/daily")
async def metrics_daily(model: str, session: Session, interval: Interval) -> list[dict[str, Any]]:
    return await queries.daily_series(session, _check_model(model), interval)


@app.get("/metrics/comparison")
async def metrics_comparison(session: Session, interval: Interval, days: Days = None) -> dict:
    """ARIMA vs GBT on the same hours: accuracy, MCC, log loss, AUC, McNemar."""
    return await queries.paired_comparison(session, interval, days)


@app.get("/metrics/by-magnitude")
async def metrics_by_magnitude(session: Session, interval: Interval, days: Days = None) -> dict:
    """§6: accuracy as a function of move size, in basis points."""
    return await queries.accuracy_by_magnitude(session, interval, days)


@app.get("/drift")
async def drift(session: Session, limit: int = Query(60, ge=1, le=1000)) -> list[dict[str, Any]]:
    return await queries.drift_history(session, limit)


@app.get("/versions")
async def versions(session: Session) -> list[dict[str, Any]]:
    return await queries.versions(session)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard(request: Request, session: Session, interval: Interval) -> HTMLResponse:
    settings = get_settings()
    context = {
        "symbol": settings.binance_symbol,
        "interval": settings.binance_interval,
        "health": await queries.health(session, settings.binance_symbol),
        "live": {
            name: [w.to_dict() for w in await queries.live_metrics(session, name, interval)]
            for name in MODEL_NAMES
        },
        "comparison": await queries.paired_comparison(session, interval, 30),
        "comparison_all": await queries.paired_comparison(session, interval, None),
        "magnitude": await queries.accuracy_by_magnitude(session, interval, None),
        "drift": await queries.drift_history(session, 60),
        "daily": {
            name: await queries.daily_series(session, name, interval) for name in MODEL_NAMES
        },
        "models": MODEL_NAMES,
    }
    return TEMPLATES.TemplateResponse(request, "dashboard.html", context)
