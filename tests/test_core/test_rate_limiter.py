import pytest
import pytest_asyncio

from app.core.exceptions import RateLimitError
from app.core.rate_limiter import RateLimiter


class _Settings:
    rate_limit_enabled = True
    rate_limit_backend = "memory"
    redis_url = ""
    default_rpm = 5
    default_tpm = 1000
    default_tpd = 100_000


@pytest_asyncio.fixture
async def limiter():
    return RateLimiter(_Settings())


@pytest.mark.asyncio
async def test_allows_within_limit(limiter):
    for _ in range(5):
        await limiter.check_and_consume("user1", None, 10)


@pytest.mark.asyncio
async def test_rpm_exceeded(limiter):
    for _ in range(5):
        await limiter.check_and_consume("user2", None, 1)
    with pytest.raises(RateLimitError):
        await limiter.check_and_consume("user2", None, 1)


@pytest.mark.asyncio
async def test_tpm_exceeded(limiter):
    with pytest.raises(RateLimitError):
        await limiter.check_and_consume("user3", None, 5000)


@pytest.mark.asyncio
async def test_disabled_no_limit():
    class Disabled:
        rate_limit_enabled = False
        default_rpm = 1
        default_tpm = 1

    lim = RateLimiter(Disabled())
    # Should never raise even beyond limits
    for _ in range(100):
        await lim.check_and_consume("userX", None, 999999)
