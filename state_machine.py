import time


class DomainStateMachine:
    IDLE        = 'IDLE'
    CHARGING    = 'CHARGING'
    ACTIVE      = 'ACTIVE'
    DEACTIVATING = 'DEACTIVATING'

    CHARGE_DURATION     = 2.0   # seconds to fully charge
    HAND_RAISE_HOLD     = 0.5   # seconds wrist must be above shoulder
    DEACTIVATE_DURATION = 1.5   # seconds for fade-out

    def __init__(self):
        self.state              = self.IDLE
        self.charge_progress    = 0.0   # 0.0 -> 1.0 during CHARGING
        self.deactivate_progress = 0.0  # 0.0 -> 1.0 during DEACTIVATING
        self.flash_frames       = 0     # countdown; main loop decrements after draw
        self._hand_raise_start  = None
        self._charge_start      = None
        self._deactivate_start  = None

    # ------------------------------------------------------------------
    def update(self, gestures: dict):
        now = time.time()

        if self.state == self.IDLE:
            self._update_idle(gestures, now)

        elif self.state == self.CHARGING:
            self._update_charging(gestures, now)

        elif self.state == self.ACTIVE:
            self._update_active(gestures, now)

        elif self.state == self.DEACTIVATING:
            self._update_deactivating(now)

    # ------------------------------------------------------------------
    def _update_idle(self, gestures, now):
        if gestures.get('hand_raised'):
            if self._hand_raise_start is None:
                self._hand_raise_start = now
            elif now - self._hand_raise_start >= self.HAND_RAISE_HOLD:
                self._transition_charging(now)
        else:
            self._hand_raise_start = None

    def _update_charging(self, gestures, now):
        elapsed = now - self._charge_start
        self.charge_progress = min(elapsed / self.CHARGE_DURATION, 1.0)

        if gestures.get('fist_clenched'):
            self._transition_active()

    def _update_active(self, gestures, now):
        if gestures.get('open_palm'):
            self._transition_deactivating(now)

    def _update_deactivating(self, now):
        elapsed = now - self._deactivate_start
        self.deactivate_progress = min(elapsed / self.DEACTIVATE_DURATION, 1.0)
        if self.deactivate_progress >= 1.0:
            self._transition_idle()

    # ------------------------------------------------------------------
    def _transition_charging(self, now):
        self.state = self.CHARGING
        self._charge_start = now
        self.charge_progress = 0.0

    def _transition_active(self):
        self.state = self.ACTIVE
        self.flash_frames = 4

    def _transition_deactivating(self, now):
        self.state = self.DEACTIVATING
        self._deactivate_start = now
        self.deactivate_progress = 0.0

    def _transition_idle(self):
        self.state = self.IDLE
        self.charge_progress = 0.0
        self.deactivate_progress = 0.0
        self.flash_frames = 0
        self._hand_raise_start = None
        self._charge_start = None
        self._deactivate_start = None

    # ------------------------------------------------------------------
    @property
    def effect_alpha(self) -> float:
        """Overall opacity multiplier for all effects (handles fade-in / fade-out)."""
        if self.state == self.CHARGING:
            return self.charge_progress
        if self.state == self.ACTIVE:
            return 1.0
        if self.state == self.DEACTIVATING:
            return 1.0 - self.deactivate_progress
        return 0.0
