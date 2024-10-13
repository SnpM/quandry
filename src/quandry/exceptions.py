class ResourceExhaustedException(Exception):
    """Exception raised when a resource is exhausted."""
    
    def __init__(self, resource_name, message="Resource has been exhausted"):
        self.resource_name = resource_name
        self.message = f"{resource_name}: {message}"
        super().__init__(self.message)