# SAML (Security Assertion Markup Language)

## Architecture
- XML-based framework for exchanging authentication and authorization data
- Used for Single Sign-On (SSO)

## Example: SAML Assertion (XML)
```xml
<saml:Assertion>
  <saml:Subject>
    <saml:NameID>user@example.com</saml:NameID>
  </saml:Subject>
  <saml:AttributeStatement>
    <saml:Attribute Name="role">
      <saml:AttributeValue>admin</saml:AttributeValue>
    </saml:Attribute>
  </saml:AttributeStatement>
</saml:Assertion>
```
