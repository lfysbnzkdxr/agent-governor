# Multi-Tenant

::: agent_governor.tenant.manager
    options:
      members:
        - TenantManager

::: agent_governor.tenant.config
    options:
      members:
        - TenantPolicy
        - MultiTenantConfig

::: agent_governor.tenant.billing
    options:
      members:
        - BillingReport
