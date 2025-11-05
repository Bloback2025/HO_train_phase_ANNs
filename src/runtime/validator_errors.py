# validator_errors placeholder
class ValidatorError(Exception):
    def __init__(self, error_code, message, remediation_hint=None, related_fields=None):
        self.error_code = error_code
        super().__init__(message)
