import time
import csv
from playwright.sync_api import Playwright, sync_playwright

# First, launch chrome with remote debugging enabled
# "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222

def run(playwright: Playwright) -> None:
    # Connect to the existing Chrome instance
    browser = playwright.chromium.connect_over_cdp("http://localhost:9222")

    # Check if there are existing contexts
    if browser.contexts:
        context = browser.contexts[0]
    else:
        context = browser.new_context()

    # Check if there are existing pages
    if context.pages:
        page = context.pages[0]
    else:
        page = context.new_page()

    # Navigate to the root website
    page.goto("https://myaeon2go.com/products/category/8412982/all")

    # Extract the main categories
    main_category_elements = page.query_selector_all('ul.cfPLiusUk3TWI4Bohy6D > li')
    category_names = []
    for element in main_category_elements:
        main_category = element.query_selector('>a').get_attribute('title')

        # Special cases
        if main_category == "Shop": # Shop all items, so no need to extract
            continue
        elif main_category == "It_&_Gadget": # Has no subcategories
            category_names.extend([(main_category, "It_&_Gadget")])
            continue

        # Extract subcategories
        subcategory_elements = element.query_selector('>div').query_selector_all('ul.OzPZZTeiowgvvZjC5DHb > li > a')
        subcategory_names = [subcategory.get_attribute('title')  for subcategory in subcategory_elements]

        # Add the main category to the subcategories unless if it is the same name (ie: Shop all in that category)
        category_names.extend([(main_category, subcategory) for subcategory in subcategory_names if subcategory != main_category])

    # Print the categories for debugging
    i = 0
    for main_category, subcategory in category_names:
        subcategory_alias = subcategory
        if main_category == "Chill_&_Frozen" and subcategory == "Non_halal":
            subcategory_alias = "Non_halal_frozen"
        elif subcategory == "Others":
            subcategory_alias = "Others_" + main_category
        elif main_category == "Non_halal":
            subcategory_alias = "Non_halal"
        
        i += 1
        print(f"{i}: {main_category} -> {subcategory_alias}")
    print()

    # Iterate through category_names and navigate to each subcategory website
    for i, (main_category, subcategory) in enumerate(category_names, 1):
        if i == 1:    
            with open('products.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csvfile.write('Product, Category, Main Category\n')
                csvfile.flush()

        # Navigate back to the root website
        page.goto("https://myaeon2go.com/products/category/8412982/all")

        # Open the menu
        page.get_by_label("Menu").click()
        page.locator("button.JjP3MoC6iogEm8iizfIT").click()

        # Navigate to the subcategory website
        if subcategory == "It_&_Gadget": # Special case, click the main category
            main_category_element = page.locator(f'ul.cfPLiusUk3TWI4Bohy6D > li > a[title="{main_category}"]')
            main_category_element.click()
        else:
            # Open main category dropdown
            main_category_element = page.locator(f'ul.cfPLiusUk3TWI4Bohy6D > li > a[title="{main_category}"]')
            main_category_element.locator('..').locator('> button.yQdexWOInKykNcuEuwrE').click()

            # Click the subcategory
            subcategory_element = main_category_element.locator('..').locator(f'ul.OzPZZTeiowgvvZjC5DHb > li > a[title="{subcategory}"]')
            subcategory_element.click()

        # Skip page if subcategory is out of stock
        time.sleep(3)
        if page.get_by_text("Oh no! Looks like this is out").is_visible():
            print(f"Skipping {main_category} -> {subcategory} as it is out of stock.")
            continue

        # Continuously scroll and click "Load More" until all items are loaded
        while True:
            try:
                # Scroll to the bottom
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                
                # Click the "Load More" button
                load_more_button = page.locator('text="LOAD MORE"')
                if not load_more_button.is_visible():
                    break
                load_more_button.click(timeout=5000)  # 5 seconds to find load more button
                time.sleep(3)
            except TimeoutError:
                print("Load More button not found in time. Likely all items are loaded.")
                break
            except Exception as e:
                print("An error occurred:", e)
                break

        # Extract product names and remove whitespace from product_names
        product_elements = page.locator('ul.g-product-list > li div.g-brand-text')
        product_names = product_elements.evaluate_all("elements => elements.map(el => el.innerHTML)")
        product_names = [name.strip() for name in product_names]

        subcategory_alias = subcategory.strip() 
        if main_category == "Chill_&_Frozen" and subcategory == "Non_halal":
            subcategory_alias = "Non_halal_frozen"
        elif subcategory == "Others":
            subcategory_alias = "Others_" + main_category
        elif main_category == "Non_halal":
            subcategory_alias = "Non_halal"

        # Print the product names for debugging
        print(subcategory_alias,":",product_names)
        print()

        # Save product names to a CSV file
        with open('products.csv', 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for name in product_names:
                writer.writerow([name, subcategory_alias, main_category])
            csvfile.flush()

        # Print the progress with a percentage of current to length
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] {i}/{len(category_names)} ({i/len(category_names)*100:.2f}%) {main_category} -> {subcategory_alias}")
        print("[" + "="*int(i/len(category_names)*50) + ">" + "."*(50-int(i/len(category_names)*50)) + "]")
        print()

    # Pause the script to keep the browser open
    input("Press Enter to continue...")

    # ---------------------
    context.close()
    browser.close()

with sync_playwright() as playwright:
    run(playwright)